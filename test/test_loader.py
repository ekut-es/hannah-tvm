#
# Copyright (c) 2022 University of Tübingen.
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

import hannah_tvm.load as load

root_dir = path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def test_tflite():
    import tflite

    model_path = (
        pathlib.Path(root_dir)
        / "models"
        / "tiny_ml_perf"
        / "keyword_spotting"
        / "kws_ref_model.tflite"
    )
    assert model_path.exists()

    with model_path.open("rb") as model_file:
        modelBuf = model_file.read()
        tflModel = tflite.Model.GetRootAsModel(modelBuf, 0)

        print(tflModel)

        assert tflModel.SubgraphsLength() == 1
        g = tflModel.Subgraphs(0)

        print(g)


if __name__ == "__main__":
    test_tflite()
