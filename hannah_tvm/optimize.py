import tvm
import tvm.relay as relay


def pre_quantize_opts(mod, param):
    trans = tvm.transform.Sequential(
        [
            relay.transform.SimplifyInference(),
            relay.transform.FoldConstant(),
            relay.transform.EliminateCommonSubexpr(),
        ]
    )

    mod = trans(mod)

    return mod 