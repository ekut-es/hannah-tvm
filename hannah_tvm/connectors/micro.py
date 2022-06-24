import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tvm
from tvm import auto_scheduler, autotvm

from .core import BoardConnector, BuildArtifactHandle, TaskConnector
from .pulp_runner import PulpRunner


class MicroTVMTaskConnector(TaskConnector):
    def __init__(self, board_config):
        self.board = board_config
        self._target = None

    def setup(self):
        self._target = tvm.target.Target(self.board.target, host=self.board.target_host)
        build_dir = Path("build")
        build_dir.mkdir(exist_ok=True)
        self.project_dir = build_dir.absolute() / f"microtvm_project"

    def target(self):
        return self._target

    def runner(self, tuner=None):
        if tuner == "autotvm":
            if self.board.rpc_runner == "pulp":
                runner = PulpRunner(Path(self.board.micro.template_dir))
            else:
                runner = autotvm.RPCRunner(
                    self.board.name,
                    host="localhost",
                    port=self._tracker_port,
                    number=5,
                    timeout=10,
                )
        elif tuner == "auto_scheduler":
            runner = auto_scheduler.RPCRunner(
                key=self.board.name, host="localhost", port=self._tracker_port
            )
        return runner

    def builder(self, tuner=None):
        runtime = tvm.relay.backend.Runtime("crt", {"system-lib": True})
        if tuner == "autotvm":
            builder = autotvm.LocalBuilder(runtime=runtime)
        elif tuner == "auto_scheduler":
            builder = "local"
        return builder

    def upload(self, mod):
        for i, m in enumerate(mod.module._collect_dso_modules()):
            (self.project_dir / "build").mkdir(exist_ok=True, parents=True)
            with open(self.project_dir / "build" / f"lib{i}.ll", "w") as file:
                file.write(m.get_source())
        return tvm.micro.generate_project(
            self.board.micro.template_dir,
            mod,
            self.project_dir,
            dict(self.board.micro.project_options),
        )

    def measure(self, handle, inputs):

        handle.build()
        handle.flash()

        with open(self.project_dir / "cycles.txt", "r") as f:
            result = f.read()
            match = re.match(r"cycles:(\d+)\n", result)
            if match:
                cycles = int(match.group(1))
                return np.array([cycles])

        return np.array([])

    def profile(self, handle, inputs):
        pass

    def teardown(self):
        pass


class MicroTVMBoardConnector(BoardConnector):
    def __init__(self, board_config):
        self._board_config = board_config

    def setup(self):
        pass

    def task_connector(self) -> TaskConnector:
        return MicroTVMTaskConnector(self._board_config)

    def is_alive(self) -> bool:
        return True

    def reset(self) -> None:
        pass

    def teardown(self):
        pass

    def boards_available(self) -> int:
        # We can always create another project
        return 1
