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
import logging
from dataclasses import dataclass
from functools import reduce

import tvm
from tvm import autotvm, relay, tir


@dataclass
class DomainElement:
    extent: int
    var: str


class MemoryAnalysis:
    def __init__(self):
        self.global_symbol = {}
        self.buffers = {}
        self.memory_accesses = []

    def _dtype_size(self, dtype):
        if dtype.startswith("int"):
            dtype = dtype.replace("int", int)
            size = int(dtype)
        elif dtype.startswith("float"):
            dtype = dtype.replace("float", "")
            size = int(dtype)
        else:
            logging.warn(f"Unknown dtype {dtype} assuming 32 bits")
            size = 32
        return size

    def _buffer_size(self, buffer):
        dtype = buffer.dtype
        element_size = self._dtype_size(dtype)
        shape = []
        if isinstance(buffer, tir.Allocate):
            shape = buffer.extents
        elif isinstance(buffer, tir.Buffer):
            shape = buffer.shape
        else:
            logging.warning(f"Unhandled buffer type: {type(buffer)}")

        elements = reduce(lambda x, y: x * y, shape, 1)
        if elements == 0:
            elements = 1

        return element_size * elements

    def extract_buffers(self, op):
        if isinstance(op, tir.stmt.Allocate):
            # print(op)
            assert op.span is None
            # if tvm.ir.structural_equal(op.condition, tvm.ir.make_node("IntImm", dtype="int32", value=1)):
            assert op.condition.value == 1
            self.buffers[op.buffer_var] = op

    def extract_memory_references(self, op, iteration_domain=None):
        if iteration_domain is None:
            iteration_domain = []

        print(type(op))

        if isinstance(op, tir.Load):
            print("load", op)
        elif isinstance(op, tir.Store):
            print("store", op)
        elif isinstance(op, tir.BufferLoad):
            logging.warning("unhandled buffer_load %s", op)
        elif isinstance(op, tir.BufferStore):
            logging.warning("buffer_store %s", op)
        elif isinstance(op, tir.ProducerLoad):
            logging.warning("producer_load %s", op)
        elif isinstance(op, tir.ProducerStore):
            logging.warning("producer_store %s", op)
        elif isinstance(op, tir.AttrStmt):
            if op.attr_key == "thread_extent":
                assert isinstance(op.node, tir.expr.IterVar)
                extent = op.value
                iter_var = op.node.thread_tag

                iteration_domain.append(DomainElement(extent, iter_var))
                print("iteration_domain", iteration_domain)

        else:
            if hasattr(op, "body"):
                self.extract_memory_references(op.body, iteration_domain)

    def __call__(self, f, mod, ctx):
        self.global_symbol = f.attrs["global_symbol"]
        for param in f.params:
            buffer = f.buffer_map[param]
            self.buffers[buffer] = buffer

        tir.stmt_functor.post_order_visit(f.body, self.extract_buffers)

        self.extract_memory_references(f.body)

        for var, buffer in self.buffers.items():
            print(var, self._buffer_size(buffer))


@tvm.tir.transform.prim_func_pass(opt_level=0)
def memory_analysis(f, mod, ctx):
    print("===================================================")
    print("Analyzing:", f)
    print("")

    analysis = MemoryAnalysis()
    analysis(f, mod, ctx)

    print("===================================================")
    print("")

    return f


def memory_main():
    import sys

    from .load import load_model

    model_file = sys.argv[1]
    mod, params, input_shapes = load_model(model_file)

    target = tvm.target.Target("nvidia/jetson-tx2")
    target_host = tvm.target.Target("llvm")

    autotvm.measure.measure_methods.set_cuda_target_arch(target.attrs["arch"])
    with tvm.transform.PassContext(
        opt_level=3, config={"tir.add_lower_pass": [(4, memory_analysis)]}
    ):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)
