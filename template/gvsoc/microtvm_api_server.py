import json
import logging
import os
import pathlib
import shutil
import subprocess
import tarfile
import typing

import tvm.micro.project_api.server as server
from tvm.relay import load_param_dict

from hannah_tvm.micro import GVSOCBuilder

logger = logging.getLogger(__name__)

HERE = pathlib.Path(__file__).parent
MODEL = "model.tar"
IS_TEMPLATE = HERE.parent.name == "template"

PROJECT_OPTIONS = [
    server.ProjectOption(
        "project_type",
        help="Type of project to generate.",
        choices=("host_driven", "aot"),
        optional=["generate_project"],
        type="str",
    ),
    server.ProjectOption(
        "compiler",
        help="Compile with gcc or clang",
        choices=("gcc", "llvm"),  # TODO(@gerum) enable tvm
        optional=["build"],
        type="str",
    ),
]


class PulpProjectAPIHandler(server.ProjectAPIHandler):

    CRT_COPY_ITEMS = ("include", "Makefile", "src")

    def server_info_query(self, tvm_version: str) -> server.ServerInfo:
        return server.ServerInfo(
            "gvsoc",
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
        logger.info("Generate project in %s", str(project_dir))
        self.model_library_format_path = model_library_format_path
        if options["project_type"] == "host_driven":
            self.generate_run_host_driven(model_library_format_path, project_dir)
        elif options["project_type"] == "static":
            self.generate_run_static(model_library_format_path, project_dir)
        elif options["project_type"] == "aot":
            self.generate_run_aot(model_library_format_path, project_dir)
        else:
            raise Exception("Unknown project type")

        shutil.copy(__file__, project_dir)

        type_dir = HERE / options["project_type"]
        if type_dir.exists():
            shutil.copytree(type_dir, project_dir, dirs_exist_ok=True)

        common_dir = HERE / "common"
        if common_dir.exists():
            shutil.copytree(common_dir, project_dir, dirs_exist_ok=True)
        global_common_dir = HERE / ".." / "common"
        if global_common_dir.exists():
            shutil.copytree(global_common_dir, project_dir, dirs_exist_ok=True)

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

    def generate_run_host_driven(self, model_library_format_path, project_dir):
        with tarfile.open(model_library_format_path) as tar:
            logger.info("model library format files:")
            for file_name in tar.getnames():
                logger.info("%s", file_name)

            tar.extractall(project_dir / "build")
            graph = json.load(tar.extractfile("./executor-config/graph/default.graph"))
            param_bytes = bytearray(
                tar.extractfile("./parameters/default.params").read()
            )
            params = load_param_dict(param_bytes)
            with open(project_dir / "build" / "runner.c", "w") as f:
                builder = GVSOCBuilder()
                builder.build(graph, params, f)

    def generate_run_static(self, model_library_format_path, project_dir):
        pass

    def generate_run_aot(self, model_library_format_path, project_dir):
        pass

    def build(self, options: dict):
        if IS_TEMPLATE:
            return
        logger.info("Building project in: %s", os.getcwd())
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

    def read_transport(self, n: int, timeout_sec: typing.Union[float, None]) -> bytes:
        return super().read_transport(n, timeout_sec)

    def open_transport(self, options: dict) -> server.TransportTimeouts:
        return super().open_transport(options)

    def close_transport(self):
        return super().close_transport()


if __name__ == "__main__":
    server.main(PulpProjectAPIHandler())
