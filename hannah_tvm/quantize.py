import tvm.relay as relay


def quantize(mod, params, calibrate_dataset=None):
    if calibrate_dataset:
        with relay.quantize.qconfig(calibrate_mode="kl_divergence", weight_scale="max"):
            mod = relay.quantize.quantize(
                mod,
                params,
                dataset=calibrate_dataset,
                skip_dense_layer=False,
                skip_conv_layers=[],
            )
    else:
        with relay.quantize.qconfig(
            calibrate_mode="global_scale",
            global_scale=128.0,
            skip_dense_layer=False,
            skip_conv_layers=[],
        ):  # , partition_conversions="enabled"):
            mod = relay.quantize.quantize(mod, params)
    return mod
