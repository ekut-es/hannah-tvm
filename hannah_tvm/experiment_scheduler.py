import logging
import time
import contextlib
import multiprocessing

from abc import ABC, abstractmethod

import tvm
import tvm.auto_scheduler as auto_scheduler
import tvm.autotvm as autotvm
import tvm.relay as relay


import tvm.rpc
import tvm.rpc.tracker
import numpy as np
import tabulate

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from . import config
from . import measure
from . import load
from .task import ModelConfig, TuningTask
from .connectors import AutomateBoardConnector


logger = logging.getLogger(__name__)


class ExperimentSchedulerBase(ABC):
    def __init__(self, config) -> None:
        self.config = config
        self.tasks = []
        self.worklist = []
        self.n_jobs = config.get("n_jobs", 0)

        self.running_tasks = {}
        logger.info("Experiment Scheduler:")
        logger.info(" Number of parallel jobs: %d", self.n_jobs)
        logger.info(" Number of tasks: %d", len(self.tasks))

        self.board_connectors = {}

    def _init_connectors(self):
        for board_name, board_config in self.config.board.items():
            connector = AutomateBoardConnector(board_config)
            connector.setup()
            self.board_connectors[board_config.name] = connector

    @abstractmethod
    def _extract_tasks(self):
        pass

    def run(self):
        self._init_connectors()
        self._extract_tasks()

        with tqdm(total=len(self.worklist)) as pbar:
            with logging_redirect_tqdm():
                while self.worklist or self.running_tasks:
                    self._restart_connections()
                    self._start_tasks()

                    for board_name, task in list(self.running_tasks.items()):
                        if not task.is_alive():
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

                    time.sleep(1.0)

        self.report()

        results = []
        for task in self.tasks:
            results.append(dict(task.results))
        return results

    def _start_tasks(self):
        if len(self.running_tasks) >= self.n_jobs and self.n_jobs != 0:
            return
        elif self.worklist:
            for idx, task in list(enumerate(self.worklist)):
                board_name = task.board_config.name

                if self.board_connectors[board_name].boards_available() > 0:
                    if board_name not in self.running_tasks:
                        self.running_tasks[board_name] = task
                        del self.worklist[idx]
                        if self.n_jobs > 0:
                            task.start()
                        else:
                            task.run()
                        break
        return

    def _restart_connections(self):
        for board_name, connector in self.board_connectors.items():
            if not connector.is_alive():
                logger.info("Connection to %s is no longer alive", board_name)
                if board_name in self.running_tasks:
                    logger.critical(
                        "Server process for %s has been terminated during tuning restarting",
                        board_name,
                    )
                    connector.reset()

    def report(self):
        results = []
        for task in self.tasks:
            results.append(task.results)

        headers = [
            "board",
            "model",
            "tuning_duration",
            "status",
            "latency",
            "latency_stdev",
        ]
        results_filtered = [
            {k: v for k, v in res.items() if k in headers} for res in results
        ]

        logging.info("Results:\n" + tabulate.tabulate(results_filtered, headers="keys"))

    def finish(self):
        for connector in self.board_connectors.values():
            connector.teardown()

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
                tune=False,
            )

            self.worklist.append(task)
            self.tasks.append(task)
