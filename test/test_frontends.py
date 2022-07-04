import pytest

try:
    import tvm
except ImportError:
    pytest.skip("TVM not available", allow_module_level=True)


from hydra import compose, initialize

import hannah_tvm.config  # noqa
from hannah_tvm.tune import main


def test_tflite():
    with initialize(config_path="../hannah_tvm/conf"):
        cfg = compose(
            config_name="config",
            overrides=["board=local_cpu", "tuner=baseline", "model=tinyml_ad01"],
        )
        main(cfg)


def test_onnx():
    with initialize(config_path="../hannah_tvm/conf"):
        cfg = compose(
            config_name="config",
            overrides=["board=local_cpu", "tuner=baseline", "model=conv-net-trax"],
        )
        main(cfg)


def test_pytorch():
    with initialize(config_path="../hannah_tvm/conf"):
        cfg = compose(
            config_name="config",
            overrides=["board=local_cpu", "tuner=baseline", "model=resnext50-224"],
        )
        main(cfg)


if __name__ == "__main__":
    test_pytorch()
