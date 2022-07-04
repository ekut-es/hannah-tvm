import contextlib
import logging
import time
from abc import ABC, abstractmethod

import numpy as np
import tabulate
import tvm.rpc.tracker
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .connectors import (
    AutomateBoardConnector,
    LocalBoardConnector,
    MicroTVMBoardConnector,
)
from .task import ModelConfig, TaskStatus, TuningTask

logger = logging.getLogger(__name__)


class ExperimentSchedulerBase(ABC):
    def __init__(self, config) -> None:
        self.config = config
        self.tasks = []
        self.worklist = []

        self.running_tasks = {}

        self.board_connectors = {}

    def _init_connectors(self):
        for board_name, board_config in self.config.board.items():
            if board_config.connector == "local":
                connector = LocalBoardConnector(board_config)
            elif board_config.connector == "micro" or (
                board_config.connector == "default" and board_config.micro
            ):
                connector = MicroTVMBoardConnector(board_config)
            elif (
                board_config.connector == "automate"
                or board_config.connector == "default"
            ):
                connector = AutomateBoardConnector(board_config)
            else:
                raise Exception(
                    "Unknown setting for board_connector on board: ", board_config.name
                )
            connector.setup()
            self.board_connectors[board_config.name] = connector

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

                    for board_name, board_connector in self.board_connectors.items():
                        if board_name not in self.running_tasks:
                            has_pending_tasks = False
                            for task in self.worklist:
                                if task.board_config.name == board_name:
                                    has_pending_tasks = True
                            if not has_pending_tasks:
                                board_connector.teardown()

        self.report()

        results = []
        for task in self.tasks:
            results.append(dict(task.results))
        return results

    def _start_tasks(self):
        if self.worklist:
            for idx, task in list(enumerate(self.worklist)):
                board_name = task.board_config.name

                if self.board_connectors[board_name].boards_available() > 0:
                    if board_name not in self.running_tasks:
                        self.running_tasks[board_name] = task
                        del self.worklist[idx]
                        task.run()
        return

    def _restart_connections(self):
        for board_name, connector in self.board_connectors.items():
            if not connector.is_alive():
                if board_name in self.running_tasks:
                    logger.info("Connection to %s is no longer alive", board_name)
                    logger.critical(
                        "Server process for %s has been terminated during tuning restarting",
                        board_name,
                    )
                    connector.reset()

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
        for connector in self.board_connectors.values():
            connector.teardown()
        self.board_connectors = []

    def __del__(self):
        self.finish()


class TuningExperimentScheduler(ExperimentSchedulerBase):
    def _extract_tasks(self):
        for board_name, board_config in self.config.board.items():
            for model_name, model_config in self.config.model.items():

                task = TuningTask(
                    board_name,
                    model_name,
                    board_config,
                    model_config=model_config,
                    task_connector=self.board_connectors[
                        board_config.name
                    ].task_connector(),
                    tuner=self.config.tuner,
                )
                self.worklist.append(task)
                self.tasks.append(task)


class BackendScheduler(ExperimentSchedulerBase):
    def __init__(self, config, model, params, task_name="backend_task"):
        super().__init__(config)

        self.model = model
        self.params = params
        self.inputs = None
        self.task_name = task_name

    @contextlib.contextmanager
    def set_inputs(self, inputs):
        self.inputs = inputs
        yield None
        self.inputs = None
        return None

    def prepare(self):
        return True

    def run(self, inputs):
        with self.set_inputs(inputs):
            super().run()

    def _extract_tasks(self):
        for board_name, board_config in self.config["board"].items():
            task = TuningTask(
                board_name,
                self.task_name,
                board_config,
                ModelConfig(self.model, self.params, self.inputs),
                tuner=self.config.tuner,
            )

            self.worklist.append(task)
            self.tasks.append(task)
