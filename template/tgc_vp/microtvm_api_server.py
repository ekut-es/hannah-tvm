#
# Copyright (c) 2022 University of Tübingen.
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
import json
import os
import pathlib
import shutil
import string
import subprocess
import tarfile
import typing

import tvm
import tvm.micro.project_api.server as server
from tvm.relay import load_param_dict

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
        choices=("gcc", "llvm"),
        optional=["build"],
        type="str",
    ),
]


class TGCProjectAPIHandler(server.ProjectAPIHandler):
    # These files and directories will be recursively copied into generated projects from the CRT.
    CRT_COPY_ITEMS = ("include", "Makefile", "src")

    def server_info_query(self, tvm_version: str) -> server.ServerInfo:
        return server.ServerInfo(
            "tgc_vp", IS_TEMPLATE, "" if IS_TEMPLATE else HERE / MODEL, PROJECT_OPTIONS
        )

    def generate_project(
        self,
        model_library_format_path: pathlib.Path,
        standalone_crt_dir: pathlib.Path,
        project_dir: pathlib.Path,
        options: dict,
    ):

        project_dir = pathlib.Path(project_dir)

        # Make project directory.and copy ourselves
        project_dir.mkdir(exist_ok=True)
        shutil.copy2(__file__, project_dir / os.path.basename(__file__))

        # Extract model
        self.model_library_format_path = model_library_format_path
        with tarfile.open(model_library_format_path) as tar:
            tar.extractall(project_dir)

        # Copy Common files
        common_path = pathlib.Path(__file__).parent / ".." / "common"
        shutil.copytree(common_path, project_dir, dirs_exist_ok=True)

        # Copy Template files
        template_path = pathlib.Path(__file__).parent / options["project_type"]
        shutil.copytree(template_path, project_dir, dirs_exist_ok=True)

        metadata_path = project_dir / "metadata.json"
        with metadata_path.open() as metadata_file:
            metadata = json.load(metadata_file)
        print("Module Metadata:")
        print("================")
        print(json.dumps(metadata, indent=4))

    def build(self, options: dict):
        if IS_TEMPLATE:
            return

        subprocess.run(
            [
                "make",
            ],
            timeout=30,
            cwd=HERE,
        )

    def flash(self, options: dict):
        if IS_TEMPLATE:
            return

        subprocess.run(
            ["make", "run", "-s"],
            timeout=180,
            cwd=HERE,
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
    server.main(TGCProjectAPIHandler())
