import pickle
from pathlib import Path

import tvm

from hannah_tvm.passes.op_order import calculate_op_order

data_dir = Path(__file__).parent / "data"


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
