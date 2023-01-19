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
from tvm import ir, relay
from tvm.relay import ExprVisitor


class OpOrder(ExprVisitor):
    def __init__(self):
        super().__init__()
        self.layers = []
        self.layer_count = 0
        self.visited = set()

    def visit_call(self, call):
        for a in call.args:
            self.visit(a)

        op = call.op
        # print(type(op))
        if isinstance(op, relay.Function):
            attrs = op.attrs
            primitive = attrs["Primitive"] if "Primitive" in attrs else None
            hash = attrs["hash"] if "hash" in attrs else None
            if primitive:
                assert hash is not None
                self.layers.append(hash)
                self.layer_count += 1

        elif isinstance(op, ir.Op):
            # print(op)
            pass
        else:
            print("Unhandled call target")

        self.visit(call.op)

    def visit_function(self, func):
        self.visit(func.body)

        for x in func.params:
            self.visit(x)


def calculate_op_order(graph):
    op_order = OpOrder()

    op_order.visit(graph)

    sorted_layers = op_order.layers

    print(sorted_layers)

    return sorted_layers
