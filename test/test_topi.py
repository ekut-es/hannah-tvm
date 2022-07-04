import pytest

try:
    import tvm
except ImportError:
    pytest.skip("TVM not available", allow_module_level=True)


import tvm.relay as relay
import tvm.te as te
import tvm.topi as topi


def get_1dconv(bw_f=8, bw_b=8, bw_w=8, bw_acc=32):
    shape_input = (1, 16, 32)
    shape_weights = (16, 16, 9)
    shape_bias = (16,)

    input = relay.var("input", shape=shape_input, dtype=f"int{bw_f}")
    weights = relay.var("weights", shape=shape_weights, dtype=f"int{bw_w}")
    bias = relay.var("bias", shape=shape_bias, dtype=f"int{bw_b}")

    conv = relay.nn.conv1d(input, weights, out_dtype=f"int{bw_acc}")
    bias_add = relay.nn.bias_add(conv, relay.cast(bias, f"int{bw_acc}"))
    requantized = relay.right_shift(bias_add, relay.const(8, dtype=f"int{bw_acc}"))
    cast = relay.cast(requantized, dtype=f"int{bw_f}")

    return relay.Function([input, weights, bias], cast)


def test_conv1d():

    conv = get_1dconv()
    print(conv)
    mod = tvm.IRModule.from_expr(conv)
    print(mod)

    target = "llvm -mtriple=riscv32-generic-eabi -mcpu=generic-rv32"
    target_host = target

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, target_host=target_host)

    lib.lib.save("lib.s")


if __name__ == "__main__":
    test_conv1d()
