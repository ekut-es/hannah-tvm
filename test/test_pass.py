import tvm
import tvm.relay as relay
import tvm.relay.testing as testing
import tvm.auto_scheduler as auto_scheduler


@tvm.tir.transform.prim_func_pass(opt_level=0)
def print_tir(f, mod, ctx):
    print("Analyzing:", f)

    return f


def test_analysis():
    mod, params = testing.resnet.get_workload(1, 10, num_layers=18)

    image_shape = (3, 224, 224)

    target = "llvm"
    target_host = "llvm"
    ctx = tvm.cpu(0)
    with tvm.transform.PassContext(
        opt_level=3, config={"tir.add_lower_pass": [(1, print_tir)]}
    ):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)


if __name__ == "__main__":
    test_analysis()
