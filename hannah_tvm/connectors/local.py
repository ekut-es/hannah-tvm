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
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import tvm
from tvm import auto_scheduler, autotvm

from .core import BoardConnector, BuildArtifactHandle, TaskConnector

logger = logging.getLogger(__name__)


@dataclass
class LocalBuildArtifactHandle(BuildArtifactHandle):
    lib: Any


class LocalTaskConnector(TaskConnector):
    def __init__(self, board_config):
        self._board_config = board_config
        self._target = None
        self.auto_scheduler_ctx = None

    def setup(self):
        self._target = tvm.target.Target(
            self._board_config.target, host=self._board_config.target_host
        )

    def target(self):
        return self._target

    def runner(self, tuner=None):
        if tuner == "autotvm":
            runner = autotvm.LocalRunner(
                enable_cpu_cache_flush=True, number=1, repeat=10
            )
        elif tuner == "auto_scheduler":
            if self.auto_scheduler_ctx is None:
                self.auto_scheduler_ctx = auto_scheduler.LocalRPCMeasureContext()
            runner = self.auto_scheduler_ctx.runner
        return runner

    def builder(self, tuner=None):
        if tuner == "autotvm":
            builder = autotvm.LocalBuilder(build_func="default")
        elif tuner == "auto_scheduler":
            builder = "local"
        return builder

    def upload(self, lib):
        # Export library
        tmp = tvm.contrib.utils.tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # Upload module to device
        return LocalBuildArtifactHandle(lib)

    def measure(self, remote_handle, inputs, reference_outputs):
        dev = self._remote_dev()
        lib = remote_handle.lib
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        logger.info("Set inputs")
        for name, val in inputs.items():
            data_tvm = tvm.nd.array(val)
            module.set_input(name, data_tvm)

        # Evaluate on Graph Executor
        logger.info("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, repeat=10, min_repeat_ms=500)
        prof_res = np.array(ftimer().results) * 1e6  # convert to microsecond

        return prof_res

    def profile(self, remote_handle, inputs):
        dev = self._remote_dev()
        # Use debug Executor to get per operator runtime
        lib = remote_handle.lib
        debug_module = tvm.contrib.debugger.debug_executor.GraphModuleDebug(
            lib["debug_create"]("default", dev), [dev], lib.get_graph_json(), None
        )
        for name, val in inputs.items():
            data_tvm = tvm.nd.array(val)
            debug_module.set_input(name, data_tvm)
        debug_profile = debug_module.profile()

        return debug_profile

    def teardown(self):
        if self.auto_scheduler_ctx is not None:
            self.auto_scheduler_ctx = None

    def _remote_dev(self):
        target = self.target()
        if str(target.kind) == "cuda":
            dev = tvm.cuda()
        else:
            dev = tvm.cpu()
        return dev


class LocalBoardConnector(BoardConnector):
    def __init__(self, board_config):
        self._board_config = board_config

    def setup(self):
        pass

    def task_connector(self):
        return LocalTaskConnector(self._board_config)

    def is_alive(self):
        return True

    def reset(self):
        pass

    def teardown(self):
        pass

    def boards_available(self):
        return 1

    def _start_tracker(self):
        pass

    def _start_server(self):
        pass
