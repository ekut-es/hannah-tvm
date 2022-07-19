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
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

from tvm.micro.build import get_standalone_crt_dir

CRT_COPY_ITEMS = ("include", "Makefile", "src")


def populate_crt(
    project_dir: Path,
    crt_items: Optional[Iterable[str]] = None,
    standalone_crt_dir: Optional[Path] = None,
):
    """Copy c runtime files to project_dir / crt

    Args:
        project_dir (Path): The target project directory
        crt_items (Optional[Iterable[str]]: the items to copy
        standalone_crt_dir (Optional[Path]): the directory containing crt files
    """

    if crt_items is None:
        crt_items = CRT_COPY_ITEMS

    if standalone_crt_dir is None:
        standalone_crt_dir = get_standalone_crt_dir()

    crt_path = project_dir / "crt"
    crt_path.mkdir()
    for item in crt_items:
        src_path = os.path.join(standalone_crt_dir, item)
        dst_path = crt_path / item
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
