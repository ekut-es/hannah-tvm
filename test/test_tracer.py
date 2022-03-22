from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn as nn
import tvm
import tvm.contrib.debugger.debug_executor
from hydra.utils import instantiate
from omegaconf import OmegaConf

torch.set_printoptions(precision=10)

try:
    from hannah.models.factory import factory
    from hannah.models.factory.pooling import ApproximateGlobalAveragePooling1D
    from hannah.models.factory.qat import (
        Conv1d,
        Conv2d,
        ConvBn1d,
        ConvBn2d,
        ConvBnReLU1d,
        ConvBnReLU2d,
        ConvReLU1d,
        ConvReLU2d,
        Linear,
    )
    from hannah.models.factory.qconfig import get_trax_qat_qconfig
    from hannah.models.factory.reduction import ReductionBlockAdd
    from hannah.models.factory.rounding import round_upward

    from hannah_tvm.tracer import (
        LegalizeQuantizedTypes,
        QuantizationTracer,
        RelayConverter,
    )
except ImportError:
    pytest.skip("hannah is not available", allow_module_level=True)


@dataclass
class Config:
    bw_b: int = 8
    bw_f: int = 8
    bw_w: int = 6
    power_of2: bool = False
    rounding_mode: str = "UPWARD"

    def get(self, name: str, default=None):
        return getattr(self, name, default)


class Cell(nn.Module):
    def __init__(self, dim=1, act=False, bw_w=8, bw_b=8, bw_f=8):
        super().__init__()
        self.qconfig = get_trax_qat_qconfig(Config(bw_w=bw_w, bw_b=bw_b, bw_f=bw_f))
        self.activation_post_process = self.qconfig.activation()
        if dim == 1:
            if act:
                self.conv = ConvBnReLU1d(
                    4,
                    8,
                    3,
                    qconfig=get_trax_qat_qconfig(
                        Config(bw_w=bw_w, bw_b=bw_b, bw_f=bw_f)
                    ),
                    padding=1,
                    bias=True,
                )
            else:
                self.conv = ConvBn1d(
                    4,
                    8,
                    3,
                    qconfig=get_trax_qat_qconfig(
                        Config(bw_w=bw_w, bw_b=bw_b, bw_f=bw_f)
                    ),
                    padding=1,
                    bias=True,
                )
            if act:
                self.conv2 = ConvReLU1d(
                    8,
                    2,
                    3,
                    qconfig=get_trax_qat_qconfig(
                        Config(bw_w=bw_w, bw_b=bw_b, bw_f=bw_f)
                    ),
                    padding=1,
                    bias=True,
                )
            else:
                self.conv2 = Conv1d(
                    8,
                    2,
                    3,
                    qconfig=get_trax_qat_qconfig(
                        Config(bw_w=bw_w, bw_b=bw_b, bw_f=bw_f)
                    ),
                    padding=1,
                    bias=True,
                )
        elif dim == 2:
            if act:
                self.conv = ConvBnReLU2d(
                    4,
                    8,
                    3,
                    qconfig=get_trax_qat_qconfig(
                        Config(bw_w=bw_w, bw_b=bw_b, bw_f=bw_f)
                    ),
                    padding=1,
                    bias=True,
                )
            else:
                self.conv = ConvBn2d(
                    4,
                    8,
                    3,
                    qconfig=get_trax_qat_qconfig(
                        Config(bw_w=bw_w, bw_b=bw_b, bw_f=bw_f)
                    ),
                    padding=1,
                    bias=True,
                )
            if act:
                self.conv2 = ConvReLU2d(
                    8,
                    2,
                    3,
                    qconfig=get_trax_qat_qconfig(
                        Config(bw_w=bw_w, bw_b=bw_b, bw_f=bw_f)
                    ),
                    padding=1,
                    bias=True,
                )
            else:
                self.conv2 = Conv2d(
                    8,
                    2,
                    3,
                    qconfig=get_trax_qat_qconfig(
                        Config(bw_w=bw_w, bw_b=bw_b, bw_f=bw_f)
                    ),
                    padding=1,
                    bias=True,
                )

    def forward(self, x):
        x = self.activation_post_process(x)
        x = self.conv(x)
        x = self.conv2(x)
        return x


def run_test(
    cell,
    input_shape,
    act,
    input_bits,
    output_bits,
    out_dtype,
    approximate=False,
    target="llvm",
):
    print(cell)
    cell.eval()
    tracer = QuantizationTracer()

    traced_graph = tracer.trace(cell)

    converter = RelayConverter(
        torch.fx.GraphModule(cell, traced_graph),
        input_scale=1 / 2 ** (input_bits - 1),
        accumulator_dtype="int20",
        input_dtype=f"int{input_bits}",
    )

    input = torch.rand(input_shape)

    mod, params = converter.run(input)
    print(mod)

    mod = tvm.relay.transform.InferType()(mod)

    mod = LegalizeQuantizedTypes()(mod)

    mod = tvm.relay.transform.InferType()(mod)
    print(mod)

    config = {}
    if target == "c":
        config = {"tir.disable_vectorize": True}

    with tvm.transform.PassContext(opt_level=3, config=config):
        lib = tvm.relay.build(mod, target=target, params=params)

    if target == "c":
        test_so_path = "test.so"
        lib.export_library(test_so_path, cc="gcc", options=["-std=c11"])
        loaded_mod = tvm.runtime.load_module(test_so_path)

        lib = loaded_mod

    input = cell.activation_post_process(input)
    output_torch = cell(input)

    input_ndarray = (input * 2 ** (input_bits - 1)).detach().numpy().astype("byte")

    dev = tvm.device(str(target), 0)
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    module.set_input("x", input_ndarray)
    module.run()
    output_scale = 1 / 2 ** (output_bits - 1)
    print("Output scale:", output_scale)
    tvm_output = (
        module.get_output(0, tvm.nd.empty(output_torch.shape, dtype=out_dtype))
        .numpy()
        .astype(float)
        * output_scale
    )

    mse = ((output_torch.detach().numpy() - tvm_output) ** 2).mean()
    max_se = ((output_torch.detach().numpy() - tvm_output) ** 2).max()

    print("Torch output:\n", output_torch)
    print("TVM output:\n", tvm_output)

    print("INT tvm output: \n", tvm_output / output_scale)
    print("INT torch output:\n", output_torch / output_scale)
    print("MSE:   ", mse)
    print("MAX_SE:", max_se)

    if approximate:
        np.testing.assert_allclose(
            tvm_output,
            output_torch.cpu().detach().numpy(),
            atol=output_scale,
            verbose=True,
        )
    else:
        np.testing.assert_allclose(
            tvm_output,
            output_torch.cpu().detach().numpy(),
            atol=output_scale,
            verbose=True,
        )
        # np.testing.assert_equal(tvm_output, output_torch.cpu().detach().numpy())


@pytest.mark.parametrize(
    "dim,act,bw_w,bw_f,bw_b,target",
    [
        (1, False, 2, 4, 8, "llvm"),
        (1, True, 2, 4, 8, "llvm"),
        (2, False, 2, 4, 8, "llvm"),
        (2, True, 2, 4, 8, "llvm"),
        (1, False, 3, 4, 8, "llvm"),
        (1, True, 3, 4, 8, "llvm"),
        (2, False, 3, 4, 8, "llvm"),
        (2, True, 3, 4, 8, "llvm"),
        (1, False, 4, 4, 8, "llvm"),
        (1, True, 4, 4, 8, "llvm"),
        (2, False, 4, 4, 8, "llvm"),
        (2, True, 4, 4, 8, "llvm"),
        (1, False, 6, 4, 8, "llvm"),
        (1, True, 6, 4, 8, "llvm"),
        (2, False, 6, 4, 8, "llvm"),
        (2, True, 6, 4, 8, "llvm"),
        (1, False, 8, 4, 8, "llvm"),
        (1, True, 8, 4, 8, "c"),
        (2, False, 8, 4, 8, "c"),
        (2, True, 8, 4, 8, "c"),
        (1, False, 2, 4, 8, "c"),
        (1, True, 2, 4, 8, "c"),
        (2, False, 2, 4, 8, "c"),
        (2, True, 2, 4, 8, "c"),
        (1, False, 3, 4, 8, "c"),
        (1, True, 3, 4, 8, "c"),
        (2, False, 3, 4, 8, "c"),
        (2, True, 3, 4, 8, "c"),
        (1, False, 4, 4, 8, "c"),
        (1, True, 4, 4, 8, "c"),
        (2, False, 4, 4, 8, "c"),
        (2, True, 4, 4, 8, "c"),
        (1, False, 6, 4, 8, "c"),
        (1, True, 6, 4, 8, "c"),
        (2, False, 6, 4, 8, "c"),
        (2, True, 6, 4, 8, "c"),
        (1, False, 8, 4, 8, "c"),
        (1, True, 8, 4, 8, "c"),
        (2, False, 8, 4, 8, "c"),
        (2, True, 8, 4, 8, "c"),
    ],
)
def test_tracer(dim, act, bw_w, bw_f, bw_b, target):
    cell = Cell(dim=dim, act=act, bw_w=bw_w, bw_f=bw_f, bw_b=bw_b)
    input_bits = bw_f
    output_bits = bw_f

    if dim == 1:
        input_shape = (1, 4, 4)
    elif dim == 2:
        input_shape = (1, 4, 4, 4)

    run_test(cell, input_shape, act, input_bits, output_bits, "int8", target=target)


class CellReduction(nn.Module):
    def __init__(self, dim=1, act=False, bw_w=8, bw_b=8, bw_f=8):
        super().__init__()
        self.qconfig = get_trax_qat_qconfig(Config(bw_w=bw_w, bw_b=bw_b, bw_f=bw_f))
        self.activation_post_process = self.qconfig.activation()
        if dim == 1:
            conv = Conv1d(
                8,
                4,
                3,
                qconfig=get_trax_qat_qconfig(Config(bw_w=bw_w, bw_b=bw_b, bw_f=bw_f)),
                padding=1,
                bias=False,
                out_quant=False,
            )

            conv2 = Conv1d(
                8,
                4,
                1,
                qconfig=get_trax_qat_qconfig(Config(bw_w=bw_w, bw_b=bw_b, bw_f=bw_f)),
                padding=0,
                bias=False,
            )
        elif dim == 2:
            conv = Conv2d(
                8,
                8,
                3,
                qconfig=get_trax_qat_qconfig(Config(bw_w=bw_w, bw_b=bw_b, bw_f=bw_f)),
                padding=1,
                bias=False,
                out_quant=False,
            )
            conv2 = Conv2d(
                8,
                8,
                3,
                qconfig=get_trax_qat_qconfig(Config(bw_w=bw_w, bw_b=bw_b, bw_f=bw_f)),
                padding=1,
                bias=False,
            )
        self.red = ReductionBlockAdd(conv, conv2)

    def forward(self, x):
        x = self.activation_post_process(x)
        x = self.red(x)
        return x


@pytest.mark.parametrize(
    "dim,act,bw_w,bw_b,bw_f",
    [
        (1, False, 4, 4, 4),
        (1, True, 7, 3, 4),
        (1, False, 2, 7, 8),
        (1, True, 8, 6, 6),
        (2, False, 3, 4, 4),
        (2, True, 7, 3, 5),
        (2, False, 2, 7, 8),
        (2, True, 8, 2, 6),
    ],
)
def test_tracer_reduction(dim, act, bw_w, bw_b, bw_f):
    cell = CellReduction(dim=dim, act=act, bw_w=bw_w, bw_b=bw_b, bw_f=bw_f)
    if dim == 1:
        input_shape = (1, 8, 32)
    elif dim == 2:
        input_shape = (1, 8, 32, 32)

    run_test(
        cell, input_shape, act, bw_f, bw_w - 1 + bw_f - 1 + 1, "int32", approximate=True
    )


class CellLinear(nn.Module):
    def __init__(self, act=False, bias=False, bw_f=8, bw_w=6, bw_b=8):
        super().__init__()
        self.qconfig = get_trax_qat_qconfig(Config(bw_f=bw_f, bw_b=bw_b, bw_w=bw_w))
        self.activation_post_process = self.qconfig.activation()
        if act:
            self.linear = LinearReLU(
                128,
                32,
                bias=bias,
                qconfig=get_trax_qat_qconfig(Config(bw_f=bw_f, bw_b=bw_b, bw_w=bw_w)),
                out_quant=True,
            )
        else:
            self.linear = Linear(
                128,
                32,
                bias=bias,
                qconfig=get_trax_qat_qconfig(Config(bw_f=bw_f, bw_b=bw_b, bw_w=bw_w)),
                out_quant=True,
            )

    def forward(self, x):
        x = self.activation_post_process(x)
        x = self.linear(x)
        return x


def test_tracer_linear(act=False, bias=False, bw_w=6, bw_b=8, bw_f=8):
    cell = CellLinear(act=act, bw_w=bw_w, bw_f=bw_f, bw_b=bw_b)
    input_shape = (1, 128)
    act = act
    run_test(cell, input_shape, act, bw_f, bw_f, "int8")


class CellPooling(nn.Module):
    def __init__(self, length=5, act=False):
        super().__init__()
        self.qconfig = get_trax_qat_qconfig(Config())
        self.activation_post_process = self.qconfig.activation()
        self.pool = ApproximateGlobalAveragePooling1D(
            length, qconfig=get_trax_qat_qconfig(Config())
        )

    def forward(self, x):
        x = self.activation_post_process(x)
        x = self.pool(x)
        return x


@pytest.mark.parametrize("length", [5, 8, 17, 33])
def test_tracer_pooling(length):
    cell = CellPooling(length=length)
    input_shape = (1, 64, length)
    act = False
    input_bits = 8
    output_bits = 8
    out_dtype = "int32"
    run_test(
        cell, input_shape, act, input_bits, output_bits, out_dtype, approximate=True
    )


class CellSimple(nn.Module):
    def __init__(self, dim=1, act=True):
        super().__init__()
        self.qconfig = get_trax_qat_qconfig(Config())
        self.activation_post_process = self.qconfig.activation()
        if dim == 1:
            if act:
                self.conv = ConvReLU1d(
                    8,
                    8,
                    3,
                    qconfig=get_trax_qat_qconfig(Config()),
                    padding=1,
                    bias=False,
                )

    def forward(self, x):
        x = self.activation_post_process(x)
        return self.conv(x)


def test_tracer_simple():
    cell = CellSimple()
    input_shape = (1, 8, 8)
    act = False
    input_bits = 8
    output_bits = 8
    out_dtype = "int8"
    run_test(cell, input_shape, act, input_bits, output_bits, out_dtype)


@pytest.mark.skip
def test_tracer_model():
    input_shape = (1, 101, 40)
    input_bits = 8
    output_bits = 15
    out_dtype = "int32"
    from pprint import pprint

    model_conf = OmegaConf.load("hannah/conf/model/conv-net-mini.yaml")
    cell = instantiate(model_conf, input_shape=input_shape, labels=12)
    cell.eval()
    print(cell)

    tracer = QuantizationTracer()

    traced_graph = tracer.trace(cell)

    converter = RelayConverter(torch.fx.GraphModule(cell, traced_graph))

    input = torch.rand(input_shape)

    mod, params = converter.run(input)
    print(mod)

    mod = tvm.relay.transform.InferType()(mod)

    mod = LegalizeQuantizedTypes()(mod)

    mod = tvm.relay.transform.InferType()(mod)
    print(mod)
    # print(params)

    target = "llvm"
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        json, lib, params = tvm.relay.build(mod, target=target, params=params)

    output_torch = cell(input)
    input_ndarray = (input * 2 ** (input_bits - 1)).detach().numpy().astype("byte")

    dev = tvm.device(str(target), 0)
    module = tvm.contrib.debugger.debug_executor.create(
        json, lib, dev, dump_root="tvmdbg"
    )
    module.set_input("x", input_ndarray)
    module.set_input(**params)
    module.run()

    output_scale = 1 / 2 ** (output_bits - 1)
    print("Output scale:", output_scale)
    tvm_output = (
        module.get_output(0, tvm.nd.empty(output_torch.shape, dtype=out_dtype))
        .numpy()
        .astype(float)
        * output_scale
    )

    mse = ((output_torch.detach().numpy() - tvm_output) ** 2).mean()
    max_se = ((output_torch.detach().numpy() - tvm_output) ** 2).max()

    print(output_torch)
    print(tvm_output)

    print("MSE:   ", mse)
    print("MAX_SE:", max_se)

    assert mse < 1 / (2 ** (min(output_bits, input_bits) - 1))
    assert max_se < 2 / (2 ** (min(output_bits, input_bits) - 1))


if __name__ == "__main__":
    test_tracer(1, True, 8, 6, 8)
    # test_tracer_linear()
    # while True:
    #    test_tracer_reduction(1, True, 8, 6, 6)
