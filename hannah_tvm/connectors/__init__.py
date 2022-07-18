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
import logging

from .automate import AutomateBoardConnector
from .core import BoardConnector
from .local import LocalBoardConnector
from .micro import MicroTVMBoardConnector

logger = logging.getLogger(__name__)


def init_board_connector(board_config) -> BoardConnector:
    logger.info("initializing board connector")
    logger.debug("Board Config: %s", str(board_config))
    if board_config.connector == "local":
        connector = LocalBoardConnector(board_config)
    elif board_config.connector == "micro" or (
        board_config.connector == "default" and board_config.micro
    ):
        connector = MicroTVMBoardConnector(board_config)
    elif board_config.connector == "automate" or board_config.connector == "default":
        connector = AutomateBoardConnector(board_config)
    else:
        raise Exception(
            "Unknown setting for board_connector on board: ", board_config.name
        )
    connector.setup()
    return connector


__all__ = [
    "init_board_connector",
    "LocalBoardConnector",
    "MicroTVMBoardConnector",
    "AutomateBoardConnector",
]
