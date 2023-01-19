#
# Copyright (c) 2023 University of Tübingen.
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


import pickle
from pathlib import Path

from hannah_tvm.passes.op_order import calculate_op_order

data_dir = Path(__file__).parent / "data"


@pytest.mark.skip("Currently not working")
def test_op_order_sine():
    sine_file = data_dir / "sine_llvm.relay.pkl"
    with sine_file.open("rb") as f:
        sine_relay = pickle.load(f)
    print(sine_relay)
    op_order = calculate_op_order(sine_relay)
    assert op_order == [
        "6a8c93f6286b00a2",
        "782b954ddd747ed9",
        "d0389ab94fe7df54",
        "218f261ff5d5d16b",
        "d0389ab94fe7df54",
        "0978af9668e587e5",
    ]


if __name__ == "__main__":
    test_op_order_sine()
