#
# Copyright (c) 2023 University of Tübingen.
#
# This file is part of hannah-tvm.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah-tvm for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import atexit
import collections
import collections.abc
import enum
import fcntl
import glob
import json
import logging
import os
import os.path
import pathlib
import queue
import re
import shlex
import shutil
import stat
import string
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
from typing import Union

import psutil
import yaml
from tvm.micro.project_api import server

_LOG = logging.getLogger(__name__)


API_SERVER_DIR = pathlib.Path(os.path.dirname(__file__) or os.path.getcwd())


BUILD_DIR = API_SERVER_DIR


MODEL_LIBRARY_FORMAT_RELPATH = "model.tar"
BOARDS = {"gem5": {}}

M3_BASE_DIR = pathlib.Path("/local/gerum/speech_recognition/external/hannah-tvm/m3")

DEFAULT_HEAP_SIZE = 128 * 1024  # Default heap size in bytes

IS_TEMPLATE = not (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH).exists()


def check_call(cmd_args, *args, **kwargs):
    cwd_str = "" if "cwd" not in kwargs else f" (in cwd: {kwargs['cwd']})"
    _LOG.info("run%s: %s", cwd_str, " ".join(shlex.quote(a) for a in cmd_args))
    return subprocess.check_call(cmd_args, *args, **kwargs)


class BoardError(Exception):
    """Raised when an attached board cannot be opened (i.e. missing /dev nodes, etc)."""


class BoardAutodetectFailed(Exception):
    """Raised when no attached hardware is found matching the board= given to ZephyrCompiler."""


PROJECT_TYPES = []
if IS_TEMPLATE:
    for d in (API_SERVER_DIR / "src").iterdir():
        if d.is_dir():
            PROJECT_TYPES.append(d.name)

PROJECT_OPTIONS = server.default_project_options(
    project_type={"choices": tuple(PROJECT_TYPES)},
    board={"choices": tuple(BOARDS.keys())},
    verbose={"optional": ["generate_project"]},
) + [
    server.ProjectOption(
        "config_main_stack_size",
        optional=["generate_project"],
        type="int",
        default=None,
        help="Sets CONFIG_MAIN_STACK_SIZE for Zephyr board.",
    ),
    server.ProjectOption(
        "heap_size_bytes",
        optional=["generate_project"],
        type="int",
        default=None,
        help="Sets the value for HEAP_SIZE_BYTES to service TVM memory allocation requests.",
    ),
    server.ProjectOption(
        "board_mem_bytes",
        optional=["generate_project"],
        type="int",
        default=None,
        help="Sets the value of total available memory, only used for sanity checking of other memory related settings",
    ),
]


class Handler(server.ProjectAPIHandler):
    def __init__(self):
        super(Handler, self).__init__()
        self._proc = None

    def server_info_query(self, tvm_version):
        return server.ServerInfo(
            platform_name="m3",
            is_template=IS_TEMPLATE,
            model_library_format_path=""
            if IS_TEMPLATE
            else (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH),
            project_options=PROJECT_OPTIONS,
        )

    # These files and directories will be recursively copied into generated projects from the CRT.
    CRT_COPY_ITEMS = ("include", "Makefile", "src")
    API_SERVER_CRT_LIBS_TOKEN = "<API_SERVER_CRT_LIBS>"
    CMAKE_ARGS_TOKEN = "<CMAKE_ARGS>"
    QEMU_PIPE_TOKEN = "<QEMU_PIPE>"

    CRT_LIBS_BY_PROJECT_TYPE = {
        "host_driven": "microtvm_rpc_server microtvm_rpc_common aot_executor_module aot_executor common",
        "aot_standalone_demo": "memory microtvm_rpc_common common",
    }

    def generate_project(
        self, model_library_format_path, standalone_crt_dir, project_dir, options
    ):
        project_type = options["project_type"]
        warning_as_error = options.get("warning_as_error")
        verbose = options.get("verbose")

        heap_size_bytes = options.get("heap_size_bytes") or DEFAULT_HEAP_SIZE
        compile_definitions = options.get("compile_definitions")
        config_main_stack_size = options.get("config_main_stack_size")
        board_mem_size = options.get("board_mem_size") or None

        project_dir = pathlib.Path(project_dir)
        # Make project directory.
        project_dir.mkdir()

        # Copy ourselves to the generated project. TVM may perform further build steps on the generated project
        # by launching the copy.
        shutil.copy2(__file__, project_dir / os.path.basename(__file__))

        # Place Model Library Format tarball in the special location, which this script uses to decide
        # whether it's being invoked in a template or generated project.
        project_model_library_format_tar_path = (
            project_dir / MODEL_LIBRARY_FORMAT_RELPATH
        )
        shutil.copy2(model_library_format_path, project_model_library_format_tar_path)

        # Extract Model Library Format tarball.into <project_dir>/model.
        extract_path = os.path.splitext(project_model_library_format_tar_path)[0]
        with tarfile.TarFile(project_model_library_format_tar_path) as tf:
            os.makedirs(extract_path)
            tf.extractall(path=extract_path)

        # Populate CRT.
        crt_path = project_dir / "crt"
        crt_path.mkdir()
        for item in self.CRT_COPY_ITEMS:
            src_path = os.path.join(standalone_crt_dir, item)
            dst_path = crt_path / item
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

            if board_mem_size is not None:
                assert (
                    heap_size_bytes < board_mem_size
                ), f"Heap size {heap_size_bytes} is larger than memory size {board_mem_size} on this board."

        # Populate crt-config.h
        crt_config_dir = project_dir / "crt_config"
        crt_config_dir.mkdir()
        shutil.copy2(
            API_SERVER_DIR / "crt_config" / "crt_config.h",
            crt_config_dir / "crt_config.h",
        )

        # Populate src/
        src_dir = project_dir / "src"
        shutil.copytree(API_SERVER_DIR / "src" / project_type, src_dir)

        # Populate extra_files
        # if extra_files_tar:
        #    with tarfile.open(extra_files_tar, mode="r:*") as tf:
        #        tf.extractall(project_dir)

        with open(API_SERVER_DIR / "Makefile.template", "r") as template_file:
            template = MakefileTemplate(template_file.read())

        srcs = (
            list(project_dir.glob("**/*.c"))
            + list(project_dir.glob("**/*.cc"))
            + list(project_dir.glob("**/*.cpp"))
        )
        objs = [str(f.with_suffix(".o")) for f in srcs]

        objs_str = " \\\n  ".join(objs)

        make_file_str = template.safe_substitute(
            {"obj_files": objs_str, "target": project_type}
        )

        with open(project_dir / "Makefile", "w") as make_file:
            make_file.write(make_file_str)

    def build(self, options):
        verbose = options.get("verbose")
        env = os.environ.copy()

        args = ["make"]

        if verbose:
            print("Building target code ...")
            print(" ".join(args))
            env["VERBOSE"] = "1"

        check_call(args, cwd=BUILD_DIR, env=env)

    def flash(self, options):
        os.makedirs("run", exist_ok=True)
        args = [
            "/local/gerum/speech_recognition/external/hannah-tvm/m3/platform/gem5/build/RISCV/gem5.opt",
            "--outdir=run",
            "--debug-file=gem5.log",
            "--debug-flags=Tcu",
            "/local/gerum/speech_recognition/external/hannah-tvm/m3/config/default.py",
            "--cpu-type DerivO3CPU",
            "--isa riscv",
            "--cmd /local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/kernel,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/tilemux,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/tilemux,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/tilemux,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/tilemux,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/tilemux,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/tilemux,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/tilemux,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/tilemux,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/tilemux,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/tilemux,build/gem5-riscv-release/bin/tilemux,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/tilemux,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/tilemux,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/tilemux,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/tilemux,"
            "--mods /local/gerum/speech_recognition/external/hannah-tvm/m3/run/boot.xml,/local/gerum/speech_recognition/external/hannah-tvm/m3/build/gem5-riscv-release/bin/root,bin/host",
            "--cpu-clock=1GHz" "--sys-clock=333MHz",
        ]

        check_call(args, cwd=BUILD_DIR)

    def open_transport(self, options):
        pass

    def close_transport(self):
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def read_transport(self, n, timeout_sec):
        if self._transport is None:
            raise server.TransportClosedError()
        return self._transport.read(n, timeout_sec)

    def write_transport(self, data, timeout_sec):
        if self._transport is None:
            raise server.TransportClosedError()

        return self._transport.write(data, timeout_sec)


# class ZephyrQemuTransport:
#     """The user-facing Zephyr QEMU transport class."""

#     def __init__(self, gdbserver_port: int = None):
#         self._gdbserver_port = gdbserver_port
#         self.proc = None
#         self.pipe_dir = None
#         self.read_fd = None
#         self.write_fd = None
#         self._queue = queue.Queue()

#     def open(self):
#         with open(BUILD_DIR / "CMakeCache.txt", "r") as cmake_cache_f:
#             for line in cmake_cache_f:
#                 if "QEMU_PIPE:" in line:
#                     self.pipe = pathlib.Path(line[line.find("=") + 1 :])
#                     break
#         self.pipe_dir = self.pipe.parents[0]
#         self.write_pipe = self.pipe_dir / "fifo.in"
#         self.read_pipe = self.pipe_dir / "fifo.out"
#         os.mkfifo(self.write_pipe)
#         os.mkfifo(self.read_pipe)

#         env = None
#         if self._gdbserver_port:
#             env = os.environ.copy()
#             env["TVM_QEMU_GDBSERVER_PORT"] = self._gdbserver_port

#         self.proc = subprocess.Popen(
#             ["ninja", "run"],
#             cwd=BUILD_DIR,
#             env=env,
#             stdout=subprocess.PIPE,
#         )
#         self._wait_for_qemu()

#         # NOTE: although each pipe is unidirectional, open both as RDWR to work around a select
#         # limitation on linux. Without this, non-blocking I/O can't use timeouts because named
#         # FIFO are always considered ready to read when no one has opened them for writing.
#         self.read_fd = os.open(self.read_pipe, os.O_RDWR | os.O_NONBLOCK)
#         self.write_fd = os.open(self.write_pipe, os.O_RDWR | os.O_NONBLOCK)

#         return server.TransportTimeouts(
#             session_start_retry_timeout_sec=2.0,
#             session_start_timeout_sec=10.0,
#             session_established_timeout_sec=10.0,
#         )

#     def close(self):
#         did_write = False
#         if self.write_fd is not None:
#             try:
#                 server.write_with_timeout(
#                     self.write_fd, b"\x01x", 1.0
#                 )  # Use a short timeout since we will kill the process
#                 did_write = True
#             except server.IoTimeoutError:
#                 pass
#             os.close(self.write_fd)
#             self.write_fd = None

#         if self.proc:
#             if not did_write:
#                 self.proc.terminate()
#             try:
#                 self.proc.wait(5.0)
#             except subprocess.TimeoutExpired:
#                 self.proc.kill()

#         if self.read_fd:
#             os.close(self.read_fd)
#             self.read_fd = None

#         if self.pipe_dir is not None:
#             shutil.rmtree(self.pipe_dir)
#             self.pipe_dir = None

#     def read(self, n, timeout_sec):
#         return server.read_with_timeout(self.read_fd, n, timeout_sec)

#     def write(self, data, timeout_sec):
#         to_write = bytearray()
#         escape_pos = []
#         for i, b in enumerate(data):
#             if b == 0x01:
#                 to_write.append(b)
#                 escape_pos.append(i)
#             to_write.append(b)

#         while to_write:
#             num_written = server.write_with_timeout(
#                 self.write_fd, to_write, timeout_sec
#             )
#             to_write = to_write[num_written:]

#     def _qemu_check_stdout(self):
#         for line in self.proc.stdout:
#             line = str(line)
#             _LOG.info("%s", line)
#             if "[QEMU] CPU" in line:
#                 self._queue.put(ZephyrQemuMakeResult.QEMU_STARTED)
#             else:
#                 line = re.sub("[^a-zA-Z0-9 \n]", "", line)
#                 pattern = r"recipe for target (\w*) failed"
#                 if re.search(pattern, line, re.IGNORECASE):
#                     self._queue.put(ZephyrQemuMakeResult.MAKE_FAILED)
#         self._queue.put(ZephyrQemuMakeResult.EOF)

#     def _wait_for_qemu(self):
#         threading.Thread(target=self._qemu_check_stdout, daemon=True).start()
#         while True:
#             try:
#                 item = self._queue.get(timeout=120)
#             except Exception:
#                 raise TimeoutError("QEMU setup timeout.")

#             if item == ZephyrQemuMakeResult.QEMU_STARTED:
#                 break

#             if item in [ZephyrQemuMakeResult.MAKE_FAILED, ZephyrQemuMakeResult.EOF]:
#                 raise RuntimeError("QEMU setup failed.")

#             raise ValueError(f"{item} not expected.")


class MakefileTemplate(string.Template):
    delimiter = "@"


if __name__ == "__main__":
    server.main(Handler())
