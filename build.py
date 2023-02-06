#
# Copyright (c) 2023 hannah-tvm contributors.
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
import subprocess
from pathlib import Path
from typing import Any, Dict


def build(setup_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(__file__).parent
    datatypes_build_dir = path / "hannah_tvm" / "datatypes"

    print("Building byod extension for arbitrary precision datatypes")
    print("setup_kwargs", setup_kwargs)
    print("Project path:", path)

    subprocess.check_call("make", cwd=datatypes_build_dir)

    return setup_kwargs
