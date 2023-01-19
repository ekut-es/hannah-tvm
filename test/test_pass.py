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
import pytest

try:
    import tvm
except ImportError:
    pytest.skip("TVM not available", allow_module_level=True)


import sys
from typing import Tuple

import tvm.auto_scheduler as auto_scheduler
import tvm.relay as relay
import tvm.relay.testing as testing


class Loop:
    def __init__(self, children, init=0, bound=1024, stride=1):
        self.init = init
        self.bound = bound
        self.stride = stride
        self.children = children

    def str(self, indent=0):
        s = (
            " " * indent
            + f"Loop[init={self.init}, bound={self.bound}, stride={self.stride}"
        )
        if self.children:
            s += "\n"
            for child in self.children:
                s += child.str(indent=indent + 2)
                s += "\n"
            s += " " * indent
        s += "]"

        return s


class Access:
    def __init__(self):
        self.offsets = []
        self.base = self.base

    def calc_addr(self, vi):
        addr = self.base

    def str(self, indent=0):
        return " " * indent + f"Access[base={self.base}"


class AnalysisTree:
    def __init__(self, children=[]):
        self.parallel = None
        self.children = children

    def iterate(self) -> Tuple[int, ...]:
        yield (0, 0, 0)

    def __str__(self):
        s = "AnalysisTree["
        if self.children:
            s += "\n"
            for child in self.children:
                s += child.str(indent=2)
                s += "\n"

        s += "]"
        return s


test_tree = AnalysisTree(
    [Loop([Loop([], bound=9)], bound=9), Loop([], bound=10, stride=2)]
)
print(test_tree)

for addr in test_tree.iterate():
    print(addr)


class MemoryAnalyzer:
    def __init__(self, linesize=64):
        self.linesize = 64

    def __call__(self, function):
        print(function)
        loops = self._extract_loops()

        buffer_access_lca = tvm.tir.analysis.detect_buffer_access_lca(function)
        workspace_bytes = tvm.tir.analysis.calculate_workspace_bytes(function, 8)
        print(workspace_bytes)

    def _extract_loops(self):
        return 0


analyzed = 0


@tvm.tir.transform.prim_func_pass(opt_level=0)
def analyze_memory(f, mod, ctx):
    global analyzed
    if analyzed == 4:
        print("Analyzing:", f.attrs["global_symbol"])
        memory_analyzer = MemoryAnalyzer()
        memory_analyzer(f)
    analyzed += 1

    return f


def test_analysis():
    mod, params = testing.resnet.get_workload(1, 10, num_layers=18)

    image_shape = (3, 224, 224)

    target = "llvm"
    target_host = "llvm"
    with tvm.transform.PassContext(
        opt_level=3, config={"tir.add_lower_pass": [(3, analyze_memory)]}
    ):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)


if __name__ == "__main__":
    test_analysis()
