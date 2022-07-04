import pytest

try:
    import tvm
except ImportError:
    pytest.skip("TVM not available", allow_module_level=True)


from hydra import compose, initialize

import hannah_tvm.config
from hannah_tvm.tune import main


def test_auto_scheduler():
    with initialize(config_path="../hannah_tvm/conf"):
        cfg = compose(
            config_name="config",
            overrides=["model=sine", "board=local_cpu", "tuner=auto_scheduler"],
        )
        main(cfg)


def test_autotvm():
    with initialize(config_path="../hannah_tvm/conf"):
        cfg = compose(
            config_name="config",
            overrides=["model=sine", "board=local_cpu", "tuner=autotvm"],
        )
        main(cfg)


if __name__ == "__main__":
    test_auto_scheduler()
    test_autotvm()
