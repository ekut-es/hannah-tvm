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

    with initialize(config_path="../hannah_tvm/conf"):
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
