#
# Copyright (c) 2023 hannah-tvm contributors.
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
import contextlib
import logging
import time
from abc import ABC, abstractmethod

import numpy as np
import tabulate
import tvm.rpc.tracker
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .connectors import init_board_connector
from .task import ModelConfig, TaskStatus, TuningTask

logger = logging.getLogger(__name__)


class ExperimentSchedulerBase(ABC):
    def __init__(self, config) -> None:
        self.config = config
        self.tasks = []
        self.worklist = []

        self.running_tasks = {}

        self.board_connector = None

    def _init_connectors(self):
        connector = init_board_connector(self.config.backend.board)
        self.board_connector = connector

    @abstractmethod
    def _extract_tasks(self):
        ...

    def run(self):
        self._init_connectors()
        self._extract_tasks()

        with tqdm(total=len(self.worklist)) as pbar:
            with logging_redirect_tqdm():
                while self.worklist or self.running_tasks:
                    self._restart_connections()
                    self._start_tasks()

                    for board_name, task in list(self.running_tasks.items()):
                        del self.running_tasks[board_name]
                        pbar.update(1)

        self.board_connector.teardown()
        self.report()

        results = []
        for task in self.tasks:
            results.append(dict(task.results))
        return results

    def _start_tasks(self):
        if self.worklist:
            for idx, task in list(enumerate(self.worklist)):
                board_name = task.board_config.name

                if self.board_connector.boards_available() > 0:
                    if board_name not in self.running_tasks:
                        self.running_tasks[board_name] = task
                        del self.worklist[idx]
                        task.run()
        return

    def _restart_connections(self):
        if not self.board_connector.is_alive():
            if self.running_tasks:
                logger.info(
                    "Connection to %s is no longer alive", self.board_config.name
                )
                logger.critical(
                    "Server process for %s has been terminated during tuning restarting",
                    self.board_config.name,
                )
                self.board_connector.reset()

    def report(self, filter="all"):
        results = []
        for task in self.tasks:
            task_results = task.results

            task_results["status"] = task.status.display_name
            task_results["tuner"] = task.tuner_config.name
            if filter == "all" or task.status == filter:
                results.append(task_results)

        headers = [
            "board",
            "model",
            "tuner",
            "tuning_duration",
            "status",
            "progress",
            "latency",
            "latency_stdev",
        ]
        results_filtered = [
            {k: v for k, v in res.items() if k in headers} for res in results
        ]
        if results:
            logging.info(
                "Results:\n" + tabulate.tabulate(results_filtered, headers="keys")
            )

    def finish(self):
        if self.board_connector is not None:
            self.board_connector.teardown()

    def __del__(self):
        self.finish()


class TuningExperimentScheduler(ExperimentSchedulerBase):
    def _extract_tasks(self):
        for model_name, model_config in self.config.model.items():
            task = TuningTask(
                model_name,
                self.config.backend.board,
                model_config=model_config,
                task_connector=self.board_connector.task_connector(),
                tuner=self.config.backend.tuner,
            )
            self.worklist.append(task)
            self.tasks.append(task)
