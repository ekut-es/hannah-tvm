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
        serial_number = options.get("serial_number")

    #         west_cmd_list = options["west_cmd"].split(" ")

    #         if _find_platform_from_cmake_file(API_SERVER_DIR / CMAKELIST_FILENAME):
    #             return  # NOTE: qemu requires no flash step--it is launched from open_transport.

    #         flash_runner = _get_flash_runner()
    #         # The nRF5340DK requires an additional `nrfjprog --recover` before each flash cycle.
    #         # This is because readback protection is enabled by default when this device is flashed.
    #         # Otherwise, flashing may fail with an error such as the following:
    #         #  ERROR: The operation attempted is unavailable due to readback protection in
    #         #  ERROR: your device. Please use --recover to unlock the device.
    #         zephyr_board = _find_board_from_cmake_file(API_SERVER_DIR / CMAKELIST_FILENAME)
    #         if zephyr_board.startswith("nrf5340dk") and flash_runner == "nrfjprog":
    #             recover_args = ["nrfjprog", "--recover"]
    #             recover_args.extend(_get_nrf_device_args(serial_number))
    #             check_call(recover_args, cwd=API_SERVER_DIR / "build")

    #         flash_extra_args = []
    #         if flash_runner == "openocd" and serial_number:
    #             flash_extra_args += ["--cmd-pre-init", f"""hla_serial {serial_number}"""]

    #         if flash_runner == "nrfjprog":
    #             flash_extra_args += _get_nrf_device_args(serial_number)

    #         check_call(
    #             west_cmd_list + ["flash", "-r", flash_runner] + flash_extra_args,
    #             cwd=API_SERVER_DIR / "build",
    #         )

    def open_transport(self, options):
        pass

    #         zephyr_board = _find_board_from_cmake_file(API_SERVER_DIR / CMAKELIST_FILENAME)
    #         emu_platform = _find_platform_from_cmake_file(API_SERVER_DIR / CMAKELIST_FILENAME)
    #         if self._is_fvp(zephyr_board, emu_platform == "armfvp"):
    #             arm_fvp_path = options["arm_fvp_path"]
    #             verbose = options.get("verbose")
    #             transport = ZephyrFvpTransport(arm_fvp_path, verbose)
    #         elif self._is_qemu(zephyr_board):
    #             gdbserver_port = options.get("gdbserver_port")
    #             transport = ZephyrQemuTransport(gdbserver_port)
    #         else:
    #             zephyr_base = options["zephyr_base"]
    #             serial_number = options.get("serial_number")
    #             transport = ZephyrSerialTransport(zephyr_base, serial_number)

    #         to_return = transport.open()
    #         self._transport = transport
    #         atexit.register(lambda: self.close_transport())
    #         return to_return

    def close_transport(self):
        pass

    #         if self._transport is not None:
    #             self._transport.close()
    #             self._transport = None

    def read_transport(self, n, timeout_sec):
        pass

    #         if self._transport is None:
    #             raise server.TransportClosedError()

    #         return self._transport.read(n, timeout_sec)

    def write_transport(self, data, timeout_sec):
        pass


#         if self._transport is None:
#             raise server.TransportClosedError()

#         return self._transport.write(data, timeout_sec)


# def _set_nonblock(fd):
#     flag = fcntl.fcntl(fd, fcntl.F_GETFL)
#     fcntl.fcntl(fd, fcntl.F_SETFL, flag | os.O_NONBLOCK)
#     new_flag = fcntl.fcntl(fd, fcntl.F_GETFL)
#     assert (new_flag & os.O_NONBLOCK) != 0, "Cannot set file descriptor {fd} to non-blocking"


# class ZephyrSerialTransport:

#     NRF5340_VENDOR_ID = 0x1366

#     # NRF5340_DK v1.0.0 uses VCOM2
#     # NRF5340_DK v2.0.0 uses VCOM1
#     NRF5340_DK_BOARD_VCOM_BY_PRODUCT_ID = {0x1055: "VCOM2", 0x1051: "VCOM1"}

#     @classmethod
#     def _lookup_baud_rate(cls, zephyr_base: str):
#         # TODO(mehrdadh): remove this hack once dtlib.py is a standalone project
#         # https://github.com/zephyrproject-rtos/zephyr/blob/v2.7-branch/scripts/dts/README.txt
#         sys.path.insert(
#             0,
#             os.path.join(zephyr_base, "scripts", "dts", "python-devicetree", "src", "devicetree"),
#         )
#         try:
#             import dtlib  # pylint: disable=import-outside-toplevel
#         finally:
#             sys.path.pop(0)

#         dt_inst = dtlib.DT(BUILD_DIR / "zephyr" / "zephyr.dts")
#         uart_baud = (
#             dt_inst.get_node("/chosen")
#             .props["zephyr,console"]
#             .to_path()
#             .props["current-speed"]
#             .to_num()
#         )
#         _LOG.debug("zephyr transport: found UART baudrate from devicetree: %d", uart_baud)

#         return uart_baud

#     @classmethod
#     def _find_nrf_serial_port(cls, serial_number: str = None):
#         com_ports = subprocess.check_output(
#             ["nrfjprog", "--com"] + _get_device_args(serial_number), encoding="utf-8"
#         )
#         ports_by_vcom = {}
#         for line in com_ports.split("\n")[:-1]:
#             parts = line.split()
#             ports_by_vcom[parts[2]] = parts[1]

#         nrf_board = usb.core.find(idVendor=cls.NRF5340_VENDOR_ID)

#         if nrf_board == None:
#             raise Exception("_find_nrf_serial_port: unable to find NRF5340DK")

#         if nrf_board.idProduct in cls.NRF5340_DK_BOARD_VCOM_BY_PRODUCT_ID:
#             vcom_port = cls.NRF5340_DK_BOARD_VCOM_BY_PRODUCT_ID[nrf_board.idProduct]
#         else:
#             raise Exception("_find_nrf_serial_port: unable to find known NRF5340DK product ID")

#         return ports_by_vcom[vcom_port]

#     @classmethod
#     def _find_openocd_serial_port(cls, serial_number: str = None):
#         return generic_find_serial_port(serial_number)

#     @classmethod
#     def _find_jlink_serial_port(cls, serial_number: str = None):
#         return generic_find_serial_port(serial_number)

#     @classmethod
#     def _find_stm32cubeprogrammer_serial_port(cls, serial_number: str = None):
#         return generic_find_serial_port(serial_number)

#     @classmethod
#     def _find_serial_port(cls, serial_number: str = None):
#         flash_runner = _get_flash_runner()

#         if flash_runner == "nrfjprog":
#             return cls._find_nrf_serial_port(serial_number)

#         if flash_runner == "openocd":
#             return cls._find_openocd_serial_port(serial_number)

#         if flash_runner == "jlink":
#             return cls._find_jlink_serial_port(serial_number)

#         if flash_runner == "stm32cubeprogrammer":
#             return cls._find_stm32cubeprogrammer_serial_port(serial_number)

#         raise RuntimeError(f"Don't know how to deduce serial port for flash runner {flash_runner}")

#     def __init__(self, zephyr_base: str, serial_number: str = None):
#         self._zephyr_base = zephyr_base
#         self._serial_number = serial_number
#         self._port = None

#     def open(self):
#         port_path = self._find_serial_port(self._serial_number)
#         self._port = serial.Serial(port_path, baudrate=self._lookup_baud_rate(self._zephyr_base))
#         return server.TransportTimeouts(
#             session_start_retry_timeout_sec=2.0,
#             session_start_timeout_sec=5.0,
#             session_established_timeout_sec=5.0,
#         )

#     def close(self):
#         self._port.close()
#         self._port = None

#     def read(self, n, timeout_sec):
#         self._port.timeout = timeout_sec
#         to_return = self._port.read(n)
#         if not to_return:
#             raise server.IoTimeoutError()

#         return to_return

#     def write(self, data, timeout_sec):
#         self._port.write_timeout = timeout_sec
#         bytes_written = 0
#         while bytes_written < len(data):
#             n = self._port.write(data)
#             data = data[n:]
#             bytes_written += n


# class ZephyrQemuMakeResult(enum.Enum):
#     QEMU_STARTED = "qemu_started"
#     MAKE_FAILED = "make_failed"
#     EOF = "eof"


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
#         _set_nonblock(self.read_fd)
#         _set_nonblock(self.write_fd)

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
#             num_written = server.write_with_timeout(self.write_fd, to_write, timeout_sec)
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


# class ZephyrFvpMakeResult(enum.Enum):
#     FVP_STARTED = "fvp_started"
#     MICROTVM_API_SERVER_INIT = "fvp_initialized"
#     MAKE_FAILED = "make_failed"
#     EOF = "eof"


# class BlockingStream:
#     """Reimplementation of Stream class from Iris with blocking semantics."""

#     def __init__(self):
#         self.q = queue.Queue()
#         self.unread = None

#     def read(self, n=-1, timeout_sec=None):
#         assert (
#             n != -1
#         ), "expect firmware to open stdin using raw mode, and therefore expect sized read requests"

#         data = b""
#         if self.unread:
#             data = data + self.unread
#             self.unread = None

#         while len(data) < n:
#             try:
#                 # When there is some data to return, fetch as much as possible, then return what we can.
#                 # When there is no data yet to return, block.
#                 data += self.q.get(block=not len(data), timeout=timeout_sec)
#             except queue.Empty:
#                 break

#         if len(data) > n:
#             self.unread = data[n:]
#             data = data[:n]

#         return data

#     readline = read

#     def write(self, data):
#         self.q.put(data)


# class ZephyrFvpTransport:
#     """A transport class that communicates with the ARM FVP via Iris server."""

#     def __init__(self, arm_fvp_path: str, verbose: bool = False):
#         self._arm_fvp_path = arm_fvp_path
#         self._verbose = verbose
#         self.proc = None
#         self._queue = queue.Queue()
#         self._import_iris()

#     def _import_iris(self):
#         assert self._arm_fvp_path, "arm_fvp_path is not defined."
#         # Location as seen in the FVP_Corstone_SSE-300_11.15_24 tar.
#         iris_lib_path = (
#             pathlib.Path(self._arm_fvp_path).parent.parent.parent / "Iris" / "Python" / "iris"
#         )

#         sys.path.insert(0, str(iris_lib_path.parent))
#         try:
#             import iris.NetworkModelInitializer
#         finally:
#             sys.path.pop(0)

#         self._iris_lib = iris

#         def _convertStringToU64Array(strValue):
#             numBytes = len(strValue)
#             if numBytes == 0:
#                 return []

#             numU64 = (numBytes + 7) // 8
#             # Extend the string ending with '\0', so that the string length is multiple of 8.
#             # E.g. 'hello' is extended to: 'hello'+\0\0\0
#             strExt = strValue.ljust(8 * numU64, b"\0")
#             # Convert the string to a list of uint64_t in little endian
#             return struct.unpack("<{}Q".format(numU64), strExt)

#         iris.iris.convertStringToU64Array = _convertStringToU64Array

#     def open(self):
#         args = ["ninja"]
#         if self._verbose:
#             args.append("-v")
#         args.append("run")
#         env = dict(os.environ)
#         env["ARMFVP_BIN_PATH"] = str(API_SERVER_DIR / "fvp-hack")
#         self.proc = subprocess.Popen(
#             args,
#             cwd=BUILD_DIR,
#             env=env,
#             stdout=subprocess.PIPE,
#         )
#         threading.Thread(target=self._fvp_check_stdout, daemon=True).start()

#         self.iris_port = self._wait_for_fvp()
#         _LOG.info("IRIS started on port %d", self.iris_port)
#         NetworkModelInitializer = self._iris_lib.NetworkModelInitializer.NetworkModelInitializer
#         self._model_init = NetworkModelInitializer(
#             host="localhost", port=self.iris_port, timeout_in_ms=1000
#         )
#         self._model = self._model_init.start()
#         self._target = self._model.get_target("component.FVP_MPS3_Corstone_SSE_300.cpu0")

#         self._target.handle_semihost_io()
#         self._target._stdout = BlockingStream()
#         self._target._stdin = BlockingStream()
#         self._model.run(blocking=False, timeout=100)
#         self._wait_for_semihost_init()
#         _LOG.info("IRIS semihosting initialized.")

#         return server.TransportTimeouts(
#             session_start_retry_timeout_sec=2.0,
#             session_start_timeout_sec=10.0,
#             session_established_timeout_sec=10.0,
#         )

#     def _fvp_check_stdout(self):
#         START_MSG = "Iris server started listening to port"
#         INIT_MSG = "microTVM Zephyr runtime - running"
#         for line in self.proc.stdout:
#             line = str(line, "utf-8")
#             _LOG.info("%s", line)
#             start_msg = re.match(START_MSG + r" ([0-9]+)\n", line)
#             init_msg = re.match(INIT_MSG, line)
#             if start_msg:
#                 self._queue.put((ZephyrFvpMakeResult.FVP_STARTED, int(start_msg.group(1))))
#             elif init_msg:
#                 self._queue.put((ZephyrFvpMakeResult.MICROTVM_API_SERVER_INIT, None))
#                 break
#             else:
#                 line = re.sub("[^a-zA-Z0-9 \n]", "", line)
#                 pattern = r"recipe for target (\w*) failed"
#                 if re.search(pattern, line, re.IGNORECASE):
#                     self._queue.put((ZephyrFvpMakeResult.MAKE_FAILED, None))

#         self._queue.put((ZephyrFvpMakeResult.EOF, None))

#     def _wait_for_fvp(self):
#         """waiting for the START_MSG to appear on the stdout"""
#         while True:
#             try:
#                 item = self._queue.get(timeout=120)
#             except Exception:
#                 raise TimeoutError("FVP setup timeout.")

#             if item[0] == ZephyrFvpMakeResult.FVP_STARTED:
#                 return item[1]

#             if item[0] in [ZephyrFvpMakeResult.MAKE_FAILED, ZephyrFvpMakeResult.EOF]:
#                 raise RuntimeError("FVP setup failed.")

#             raise ValueError(f"{item} not expected.")

#     def _wait_for_semihost_init(self):
#         """waiting for the INIT_MSG to appear on the stdout"""
#         while True:
#             try:
#                 item = self._queue.get(timeout=240)
#             except Exception:
#                 raise TimeoutError("semihost init timeout.")

#             if item[0] == ZephyrFvpMakeResult.MICROTVM_API_SERVER_INIT:
#                 return

#             raise ValueError(f"{item} not expected.")

#     def close(self):
#         self._model._shutdown_model()
#         self._model.client.disconnect(force=True)
#         parent = psutil.Process(self.proc.pid)
#         if parent:
#             for child in parent.children(recursive=True):
#                 child.terminate()
#             parent.terminate()

#     def read(self, n, timeout_sec):
#         return self._target.stdout.read(n, timeout_sec)

#     def write(self, data, timeout_sec):
#         self._target.stdin.write(data)


class MakefileTemplate(string.Template):
    delimiter = "@"


if __name__ == "__main__":
    server.main(Handler())
