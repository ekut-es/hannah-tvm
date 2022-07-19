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
import tvm
import tvm.autotvm as autotvm
import tvm.te as te
from tvm.topi.pulp import conv2d_nhwc_ohwi, schedule_conv2d_nhwc_ohwi


def test_example():

    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    k = te.reduce_axis((10, n), "k")
    C = te.compute((1,), lambda _: te.sum(A[k] * B[k], axis=k), name="C")

    s = te.create_schedule(C.op)
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    print("---------cutting line---------")
    s = s.normalize()
    print(tvm.lower(s, [A, B, C], simple_mode=True))


def test_simple_schedule():

    b = te.var("batch")
    o = te.var("out_height")
    input = te.placeholder((b, o), name="input")

    output = te.compute((b, o), lambda x, y: input[x, y] * input[x, y], name="compute")

    target = tvm.target.Target(
        "llvm -mtriple=riscv32 -mcpu=generic-rv32 -mattr=+m,+c,+xpulpv -keys=pulp -runtime=c -system-lib=1 --link-params=1"
    )

    with target:

        s = te.create_schedule([output.op])
        print(tvm.lower(s, [input], simple_mode=True))


def test_simple_conv2d():
    in_dtype = "int8"
    out_dtype = "int32"

    batch = te.var("batch")
    out_height = te.var("out_height")
    out_width = te.var("out_width")
    out_channel = te.var("out_channel")
    in_channel = 8

    input = te.placeholder(
        (batch, out_height, out_width, in_channel), name="input", dtype="int8"
    )
    filter = te.placeholder(
        (out_channel, 1, 1, in_channel), name="filter", dtype="int8"
    )

    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, 1), name="ry")
    rx = te.reduce_axis((0, 1), name="rx")

    output = te.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: te.sum(
            input[nn, yy + ry, xx + rx, rc].astype(out_dtype)
            * filter[ff, ry, rx, rc].astype(out_dtype),
            axis=[ry, rx, rc],
        ),
        name="Conv2dOutput",
        tag="conv2d_nhwc_ohwi",
    )

    target = tvm.target.Target(
        "llvm -mtriple=riscv32 -mcpu=generic-rv32 -mattr=+m,+c,+xpulpv -keys=pulp -runtime=c -system-lib=1 --link-params=1"
    )
    with target:
        s = te.create_schedule([output.op])
        print(tvm.lower(s, [input, filter], simple_mode=True))


def test_conv2d_nhwc_ohwi():
    target = tvm.target.Target(
        "llvm -mtriple=riscv32 -mcpu=generic-rv32 -mattr=+m,+c,+xpulpv -keys=pulp -runtime=c -system-lib=1 --link-params=1"
    )

    with target:
        act = te.placeholder((1, 32, 32, 14), name="act", dtype="int8")
        weight = te.placeholder((32, 3, 3, 14), name="weight", dtype="int8")
        conv = conv2d_nhwc_ohwi(act, weight, 2, 0, 1, "int32")
        print("=====")
        print(conv.op)

        print("---default schedule---")
        s = te.create_schedule([conv.op])
        print(tvm.lower(s, [act, weight], simple_mode=True))

        print("----schedule---")
        schedule = schedule_conv2d_nhwc_ohwi(conv)

        print(tvm.lower(schedule, [act, weight], simple_mode=False))

        with tvm.transform.PassContext(opt_level=3):
            module = tvm.build(schedule, [act, weight], target=target, name="conv2d")
        print(module.get_source())
        # breakpoint()
        print(module)


if __name__ == "__main__":
    # test_example()
    # test_simple_schedule()
    test_conv2d_nhwc_ohwi()
    # test_simple_conv2d()
