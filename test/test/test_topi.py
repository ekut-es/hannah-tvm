import tvm
import tvm.topi as topi
import tvm.te as te
import tvm.relay as relay

def get_1dconv(bw_f=8, bw_b=8, bw_w=8, bw_acc=32):
    shape_input = (1, 16, 32)
    shape_weights = (16, 16, 9)
    shape_bias = (16,)

    input = te.placeholder(shape_input, dtype=f"int{bw_f}", name="weights")
    weights = te.placeholder(shape_weights, dtype=f"int{bw_w}", name="weights")
    bias = te.placeholder(shape_bias, dtype=f"int{bw_b}", name="bias")

    conv = relay.nn.conv1d(input, weights, out_dtype=f"int{bw_acc}")
    bias_add = relay.nn.bias_add(conv, bias)
    requantized = relay.right_shift(bias_add, 8)

    return requantized


def test_conv1d():

    conv = get_1dconv()
    print(conv)


if __name__ == "__main__":
    test_conv1d()
