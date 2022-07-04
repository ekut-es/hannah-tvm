import pytest

try:
    import tvm
except ImportError:
    pytest.skip("TVM not available", allow_module_level=True)


import logging
import os

import numpy as np
import pytest
from tvm import micro, te, topi


def run_conv2d_bitparallel(B, C, H, W, conv_size, bits, target):
    max_value = 2 ^ (bits - 1) - 1
    min_value = -2 ^ (bits - 1)

    input = np.random.randint(min_value, max_value, (B, C, H, W))
    weight = np.random.randint(min_value, max_value, (C, C, conv_size, conv_size))

    ph_input = te.placeholder(input.shape, name="input", dtype=f"int{bits}")
    ph_weight = te.placeholder(weight.shape, name="weight", dtype=f"int{bits}")

    conv = topi.nn.conv2d(
        ph_input,
        ph_weight,
        strides=(1, 1),
        padding=(conv_size // 2, conv_size // 2),
        dilation=(0, 0),
    )

    s_conv = te.create_schedule(conv.op)
    print(tvm.lower(s_conv, [ph_input, ph_weight], simple_mode=True))

    with tvm.transform.PassContext(opt_level=3):
        tvm.build()

    target = tvm.target.Target(f"{target}")
    compiler = tvm.micro.DefaultCompiler(target=target)
    opts = tvm.micro.default_options(
        os.path.join(tvm.micro.get_standalone_crt_dir(), "template", "host")
    )


if __name__ == "__main__":
    run_conv2d_bitparallel(1, 8, 24, 24, 3, bits=4, target="llvm")
