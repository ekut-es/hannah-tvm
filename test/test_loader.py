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

try:
    import tvm
except ImportError:
    pytest.skip("TVM not available", allow_module_level=True)


import os
import pathlib

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

import hannah_tvm.config as config
import hannah_tvm.load as load

root_dir = path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


models = []
config_path = pathlib.Path("..") / "hannah_tvm" / "conf"
with initialize(config_path=str(config_path), version_base=None):
    cfg = compose(config_name="config")
    for model_name, model in cfg.model.items():
        models.append((model_name, model))


@pytest.mark.parametrize("model_name,model_cfg", models)
def test_loader(model_name, model_cfg):
    print("")
    print("Testing loading of:", model_name)
    print(model_cfg)
    print("")

    assert "url" in model_cfg
    assert "filename" in model_cfg
    assert "input_shapes" in model_cfg

    model = load.load_model(model_cfg)

    print("model:", model)
