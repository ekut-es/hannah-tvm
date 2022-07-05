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
