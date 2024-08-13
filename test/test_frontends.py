#
# Copyright (c) 2024 hannah-tvm contributors.
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
import pytest
from importlib_metadata import version

try:
    import tvm
except ImportError:
    pytest.skip("TVM not available", allow_module_level=True)


from hydra import compose, initialize

import hannah_tvm.config  # noqa
from hannah_tvm.tune import main


@pytest.mark.xfail(reason="models have been removed")
def test_tflite():
    with initialize(config_path="../hannah_tvm/conf", version_base="1.2"):
        cfg = compose(
            config_name="config",
            overrides=[
                "backend/board=local_cpu",
                "backend/tuner=baseline",
                "model=tinyml_ad01",
            ],
        )
        main(cfg)


@pytest.mark.xfail(reason="models have been removed")
def test_onnx():
    with initialize(config_path="../hannah_tvm/conf", version_base="1.2"):
        cfg = compose(
            config_name="config",
            overrides=[
                "backend/board=local_cpu",
                "backend/tuner=baseline",
                "model=conv-net-trax",
            ],
        )
        main(cfg)


@pytest.mark.xfail(reason="models have been removed")
def test_pytorch():
    with initialize(config_path="../hannah_tvm/conf", version_base="1.2"):
        cfg = compose(
            config_name="config",
            overrides=[
                "backend/board=local_cpu",
                "backend/tuner=baseline",
                "model=resnext50-224",
            ],
        )
        main(cfg)


if __name__ == "__main__":
    test_pytorch()
