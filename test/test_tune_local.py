#
# Copyright (c) 2023 hannah-tvm contributors.
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


from hydra import compose, initialize

import hannah_tvm.config
from hannah_tvm.tune import main


def test_auto_scheduler():
    with initialize(config_path="../hannah_tvm/conf", version_base="1.2"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=sine",
                "backend/board=local_cpu",
                "backend/tuner=auto_scheduler",
            ],
        )
        main(cfg)


def test_autotvm():
    with initialize(config_path="../hannah_tvm/conf", version_base="1.2"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=sine",
                "backend/board=local_cpu",
                "backend/tuner=autotvm",
            ],
        )
        main(cfg)


if __name__ == "__main__":
    test_auto_scheduler()
    test_autotvm()
