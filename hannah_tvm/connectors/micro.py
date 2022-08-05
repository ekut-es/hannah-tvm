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
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tvm
from tvm import auto_scheduler, autotvm

from ..micro.gvsoc_runner import GVSOCRunner
from .core import BoardConnector, BuildArtifactHandle, TaskConnector


class MicroTVMTaskConnector(TaskConnector):
    def __init__(self, board_config):
        self.board = board_config
        self._target = None

    def setup(self):
        self._target = tvm.target.Target(self.board.target, host=self.board.target_host)
        build_dir = Path("build")
        build_dir.mkdir(exist_ok=True)
        self.project_dir = build_dir.absolute()

    def target(self):
        return self._target

    def runner(self, tuner=None):
        if tuner == "autotvm":
            if self.board.rpc_runner == "gvsoc":
                runner = GVSOCRunner(
                    Path(self.board.micro.template_dir) / "host_driven"
                )
            else:
                raise Exception("Autotuning is not supported on board")

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

        if self.board.rpc_runner == "gvsoc":
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
