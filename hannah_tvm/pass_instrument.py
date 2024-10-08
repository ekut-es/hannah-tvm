#
# Copyright (c) 2024 hannah-tvm contributors.
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
from typing import Sequence, Union

import tvm


@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def __init__(self, names: Union[str, Sequence[str]] = []):
        self.names = names

    def run_after_pass(self, mod, info):
        if (
            self.names == "all"
            or str(info.name) in self.names
            or str(info.name) == self.names
        ):
            print("Mod after pass:", info.name)
            print(mod)
