import json
import os
import pathlib
import shutil
import subprocess
import tarfile
import typing

import tvm.micro.project_api.server as server
from tvm.relay import load_param_dict

try:
    from relay_runner import runner
except:
    pass

HERE = pathlib.Path(__file__).parent
MODEL = "model.tar"
IS_TEMPLATE = HERE.parent.name == "template"

PROJECT_OPTIONS = [
    server.ProjectOption(
        "project_type",
        help="Type of project to generate.",
        choices=("host_driven",),
        optional=["generate_project"],
        type="str",
    ),
    server.ProjectOption(
        "compiler",
        help="Compile with gcc or clang",
        choices=("gcc", "llvm"),
        optional=["build"],
        type="str",
    ),
]


class PulpProjectAPIHandler(server.ProjectAPIHandler):

    CRT_COPY_ITEMS = ("include", "Makefile", "src")

    def server_info_query(self, tvm_version: str) -> server.ServerInfo:
        return server.ServerInfo(
            "pulpissimo",
            IS_TEMPLATE,
            "" if IS_TEMPLATE else HERE / MODEL,
            PROJECT_OPTIONS,
        )

    def generate_project(
        self,
        model_library_format_path: pathlib.Path,
        standalone_crt_dir: pathlib.Path,
        project_dir: pathlib.Path,
        options: dict,
    ):

        project_tvm_runtime_dir = project_dir / "tvm" / "runtime"

        self.model_library_format_path = model_library_format_path
        with tarfile.open(model_library_format_path) as tar:
            tar.extractall(project_dir / "build")
            graph = json.load(tar.extractfile("./executor-config/graph/graph.json"))
            param_bytes = bytearray(
                tar.extractfile("./parameters/default.params").read()
            )
            params = load_param_dict(param_bytes)
            with open(project_dir / "build" / "runner.c", "w") as f:
                runner(graph, params, f)

        shutil.copy(__file__, project_dir)

        shutil.copy(HERE / "Makefile", project_dir)
        shutil.copy(HERE / "runner.h", project_dir)
        shutil.copy(HERE / "test.c", project_dir)
        shutil.copy(HERE / "utvm_runtime_api.c", project_dir)
        shutil.copy(HERE / "utvm_runtime_api.h", project_dir)

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

        if os.path.exists("/tmp/utvm_project"):
            shutil.rmtree("/tmp/utvm_project")
        shutil.copytree(project_dir, "/tmp/utvm_project")

    def build(self, options: dict):
        if IS_TEMPLATE:
            return
        subprocess.run(
            [
                "make",
                "conf",
                "clean",
                "all",
                f"CONFIG_OPT='compiler={options['compiler']}'",
            ],
            timeout=30,
            cwd=HERE,
        )

    def flash(self, options: dict):
        if IS_TEMPLATE:
            return
        with open(HERE / "cycles.txt", "wb") as out:
            subprocess.run(
                ["make", "run", "-s", f"CONFIG_OPT='compiler={options['compiler']}'"],
                timeout=30,
                cwd=HERE,
                stdout=out,
            )

    def write_transport(self, data: bytes, timeout_sec: float):
        return super().write_transport(data, timeout_sec)

    def read_transport(
        self, n: int, timeout_sec: typing.Union[float, type(None)]
    ) -> bytes:
        return super().read_transport(n, timeout_sec)

    def open_transport(self, options: dict) -> server.TransportTimeouts:
        return super().open_transport(options)

    def close_transport(self):
        return super().close_transport()


if __name__ == "__main__":
    server.main(PulpProjectAPIHandler())
