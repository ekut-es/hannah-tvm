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
import time
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import tvm
import tvm.auto_scheduler as auto_scheduler
import tvm.autotvm as autotvm
import tvm.rpc as rpc

from .automate_server import AutomateServer, automate_context
from .core import BoardConnector, BuildArtifactHandle, TaskConnector

logger = logging.getLogger(__name__)


@dataclass
class AutomateBuildArtifactHandle(BuildArtifactHandle):
    remote: Any
    rlib: Any
    lib: Any


class AutomateTaskConnector(TaskConnector):
    def __init__(self, board_config, tracker_port):
        self._board_config = board_config
        self._tracker_port = tracker_port
        self._target = None

    def setup(self):
        self._target = tvm.target.Target(
            self._board_config.target, host=self._board_config.target_host
        )

    def target(self):
        return self._target

    def runner(self, tuner=None):
        if tuner == "autotvm":
            runner = autotvm.RPCRunner(
                self._board_config.name,
                host="localhost",
                port=self._tracker_port,
                number=1,
                repeat=10,
                enable_cpu_cache_flush=True,
                timeout=10,
            )
        elif tuner == "auto_scheduler":
            runner = auto_scheduler.RPCRunner(
                key=self._board_config.name, host="localhost", port=self._tracker_port
            )
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
        logger.info("Upload...")
        remote = auto_scheduler.utils.request_remote(
            self._board_config.name, "localhost", self._tracker_port, timeout=10000
        )

        remote.upload(tmp.relpath(filename))

        rlib = remote.load_module(filename)
        logger.info("Upload finished")
        return AutomateBuildArtifactHandle(remote, rlib, lib)

    def measure(self, remote_handle, inputs, reference_outputs):
        dev = self._remote_dev(remote_handle.remote)
        rlib = remote_handle.rlib
        lib = remote_handle.lib
        module = tvm.contrib.graph_executor.GraphModule(rlib["default"](dev))
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
        dev = self._remote_dev(remote_handle.remote)
        # Use debug Executor to get per operator runtime
        rlib = remote_handle.rlib
        lib = remote_handle.lib
        debug_module = tvm.contrib.debugger.debug_executor.GraphModuleDebug(
            rlib["debug_create"]("default", dev), [dev], lib.get_graph_json(), None
        )
        for name, val in inputs.items():
            data_tvm = tvm.nd.array(val)
            debug_module.set_input(name, data_tvm)
        debug_profile = debug_module.profile()

        return debug_profile

    def teardown(self):
        pass

    def _remote_dev(self, remote):
        target = self.target()
        if str(target.kind) == "cuda":
            dev = remote.cuda()
        else:
            dev = remote.cpu()
        return dev


class AutomateBoardConnector(BoardConnector):
    def __init__(self, board_config):
        self._board_config = board_config

        self._tracker_port: Union[int, None] = None
        self._tracker_conn = None
        self._tracker = None

        self._server_process: Union[AutomateServer, None] = None

    def setup(self):
        self._start_tracker()
        self._server_process = AutomateServer(self._board_config, self._tracker_port)
        self._server_process.start()

    def task_connector(self):
        connector = AutomateTaskConnector(self._board_config, self._tracker_port)
        return connector

    def is_alive(self):
        if not self._server_process.is_alive():
            return False
        return True

    def reset(self):
        self._server_process = AutomateServer(self._board_config, self._tracker_port)
        self._server_process.start()

    def teardown(self):
        self._server_process.finish()
        board = automate_context().board(self._board_config.name)
        board.unlock()

    def boards_available(self):
        board_summary = self._tracker_conn.summary()
        queue_summary = board_summary["queue_info"]
        board_name = self._board_config.name
        if board_name in queue_summary:
            return queue_summary[board_name]["free"]
        return 0

    def _start_tracker(self):
        """Start tvm remote tracker"""
        logger.info("Starting experiment tracker")
        host = "0.0.0.0"
        self._tracker = rpc.tracker.Tracker(
            host, port=9000, port_end=9090, silent=False
        )
        time.sleep(1.0)
        self._tracker_port = self._tracker.port
        self._tracker_conn = rpc.connect_tracker("localhost", self._tracker.port)

    def _start_server(self):
        """Start connection to server process"""
        board_config = self._board_config
        board_name = board_config.name
        logger.info("Starting server for %s", board_name)
        current_automate_context = automate_context()
        board = current_automate_context.board(board_name)
        status = board.trylock()
        if status:
            server_process = AutomateServer(board_config, self._tracker_port)
            self.server = server_process
            server_process.start()
