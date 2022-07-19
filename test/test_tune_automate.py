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


import warnings

from hydra import compose, initialize
from pytest import mark

import hannah_tvm.config  # noqa
from hannah_tvm.connectors.automate_server import automate_available, automate_context
from hannah_tvm.tune import main


@mark.parametrize(
    "board,tuner,model",
    [("jetsontx2_cpu", "autotvm", "sine"), ("jetsontx2_cpu", "auto_scheduler", "sine")],
)
def test_auto_scheduler(board, tuner, model):
    if not automate_available:
        warnings.warn("Skipping automate tests as automate is not available")
        return

    with initialize(config_path="../hannah_tvm/conf", version_base="1.2"):
        cfg = compose(
            config_name="config",
            overrides=[f"model={model}", f"board={board}", f"tuner={tuner}"],
        )

        current_automate_context = automate_context()

        locked_boards = []
        try:
            have_lock = False
            for id, config in cfg.board.items():
                try:
                    # Try to get board to check availability
                    current_automate_context.board(config.name)
                except Exception:
                    have_lock = False
                    break

                have_lock = current_automate_context.board(config.name).trylock()
                if not have_lock:
                    break
                locked_boards.append(config.name)

            if have_lock:
                main(cfg)
        finally:
            for board_name in locked_boards:
                current_automate_context.board(board_name).unlock()
