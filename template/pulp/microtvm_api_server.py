import tvm.micro.project_api.server as server

import pathlib
import typing
import tarfile
import json
import shutil
import os
import subprocess
try:
    from relay_runner import runner
except:
    pass

HERE = pathlib.Path(__file__).parent
MODEL = "model.tar"
IS_TEMPLATE = HERE.parent.name == "template"

PROJECT_OPTIONS=[
    server.ProjectOption(
        "project_type",
        help="Type of project to generate.",
        choices=("host_driven",),
        optional=["generte_project"],
        type="str"
    ),
    server.ProjectOption(
        "compiler",
        help="Compile with gcc or clang",
        choices=("gcc", "clang"),
        optional=["build"],
        type="str"
    )
]

class PulpProjectAPIHandler(server.ProjectAPIHandler):

    def server_info_query(self, tvm_version: str) -> server.ServerInfo:
        return server.ServerInfo(
            "pulpissimo",
            IS_TEMPLATE,
            "" if IS_TEMPLATE else HERE / MODEL,
            PROJECT_OPTIONS)

    def generate_project(self, model_library_format_path: pathlib.Path, standalone_crt_dir: pathlib.Path, project_dir: pathlib.Path, options: dict):

        self.model_library_format_path = model_library_format_path
        with tarfile.open(model_library_format_path) as tar:
            tar.extractall(project_dir / "build")
            graph = json.load(tar.extractfile("./executor-config/graph/graph.json"))
            with open(project_dir / "build" / "runner.c", "w") as f:
                runner(graph, f)

        shutil.copy(__file__, project_dir)

        shutil.copy(HERE / "Makefile", project_dir)
        shutil.copy(HERE / "runner.h", project_dir)
        shutil.copy(HERE / "test.c", project_dir)
        shutil.copy(HERE / "utvm_runtime_api.c", project_dir)
        shutil.copy(HERE / "utvm_runtime_api.h", project_dir)



    def build(self, options: dict):
        if IS_TEMPLATE:
            return
        subprocess.run(["make", "conf", "clean", "all"], timeout=30, cwd=HERE)

    def flash(self, options: dict):
        if IS_TEMPLATE:
            return
        with open(HERE / "cycles.txt", "wb") as out:
            subprocess.run(["make", "run", "-s"], timeout=30, cwd=HERE, stdout=out)

    def write_transport(self, data: bytes, timeout_sec: float):
        return super().write_transport(data, timeout_sec)

    def read_transport(self, n: int, timeout_sec: typing.Union[float, type(None)]) -> bytes:
        return super().read_transport(n, timeout_sec)

    def open_transport(self, options: dict) -> server.TransportTimeouts:
        return super().open_transport(options)

    def close_transport(self):
        return super().close_transport()

if __name__ == "__main__":
    server.main(PulpProjectAPIHandler())