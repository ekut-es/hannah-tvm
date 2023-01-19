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
import tvm
import tvm.tir as tir


class OpCounter:
    def __init__(self):
        self.counts = 0

    def __call__(self, f):
        return self.count(f)

    def count(self, f):
        print(f.attrs)


def op_counter():
    counter = OpCounter()

    return tvm.tir.transform.prim_func_pass(
        counter, opt_level=0, name="hannah_tvm.tir.op_counter"
    )
