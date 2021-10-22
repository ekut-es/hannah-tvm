import logging
import time
import contextlib

import tvm
import tvm.auto_scheduler as auto_scheduler
import tvm.autotvm as autotvm
import tvm.relay as relay
import tvm.rpc
import tvm.rpc.tracker
import numpy as np

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from automate import AutomateContext, AutomateConfig

from . import config
from . import measure
from . import load
from .task import ModelConfig, TuningTask

logger = logging.getLogger(__name__)


class ExperimentSchedulerBase:
    def __init__(self, config) -> None:
        self.config = config
        self.tasks = []
        self.worklist = []
        self.n_jobs = config.get("n_jobs", 4)
        self.running_tasks = {}

        logger.info("Starting experiment tracker")
        host = "0.0.0.0"
        self.tracker = tvm.rpc.tracker.Tracker(
            host, port=9000, port_end=9090, silent=False
        )
        time.sleep(1.0)
        self.tracker_port = self.tracker.port

        self.tracker_conn = tvm.rpc.connect_tracker("localhost", self.tracker.port)

        self._automate_config = AutomateConfig()
        self._automate_context = AutomateContext(self._automate_config)

        self.server = {}

    def _extract_tasks(self):
        pass

    def run(self):
        self._extract_tasks()

        with tqdm(total=len(self.worklist)) as pbar:
            with logging_redirect_tqdm():
                while self.worklist or self.running_tasks:
                    for board_name, server in list(self.server.items()):
                        if not server.is_alive():
                            logger.info(
                                "Server process for %s is no longer alive removing from list of servers",
                                board_name,
                            )
                            del self.server[board_name]
                            logger.info(str(self.server))
                            if board_name in self.running_tasks:
                                logger.critical(
                                    "Server process for %s has been terminated during tuning restarting",
                                    board_name,
                                )
                                task = self.running_tasks[board_name]
                                self._start_server(task.board_config)

                    board_summary = self.tracker_conn.summary()

                    if len(self.running_tasks) >= self.n_jobs and self.n_jobs != 0:
                        pass
                    elif self.worklist:
                        for idx, task in list(enumerate(self.worklist)):
                            board_name = task.board_config.name
                            if not (board_name in self.server):
                                self._start_server(task.board_config)
                            else:
                                queue_summary = board_summary["queue_info"]
                                if (
                                    board_name in queue_summary
                                    and queue_summary[board_name]["free"] > 0
                                ):
                                    if board_name not in self.running_tasks:
                                        self.running_tasks[board_name] = task
                                        del self.worklist[idx]
                                        if self.n_jobs > 0:
                                            task.start()
                                        else:
                                            task.run()
                                        break

                    for board_name, task in list(self.running_tasks.items()):
                        if not task.is_alive():
                            del self.running_tasks[board_name]
                            pbar.update(1)

                    for board_name, server_process in self.server.items():
                        if board_name not in self.running_tasks:
                            has_pending_tasks = False
                            for task in self.worklist:
                                if task.board_config.name == board_name:
                                    has_pending_tasks = True
                            if not has_pending_tasks:
                                server_process.finish()
                                board = self._automate_context.board(board_name)
                                board.unlock()

                    time.sleep(1.0)

        self.report()

        results = []
        for task in self.tasks:
            results.append(dict(task.results))
        return results

    def _start_server(self, board_config):
        board_name = board_config.name
        logger.info("Starting server for %s", board_name)
        board = self._automate_context.board(board_name)
        status = board.trylock()
        if status:
            server_process = measure.ServerProcess(board_config, self.tracker_port)
            self.server[board_name] = server_process
            server_process.start()

    def report(self):
        import tabulate

        results = []
        for task in self.tasks:
            results.append(task.results)

        logging.info("Results:\n" + tabulate.tabulate(results))

    def finish(self):
        if self.tracker is not None:
            if hasattr(self.tracker, "proc"):
                self.tracker.terminate()
            self.tracker = None

    def __del__(self):
        self.finish()


class TuningExperimentScheduler(ExperimentSchedulerBase):
    def _extract_tasks(self):
        for board_name, board_config in self.config.board.items():
            for model_name, network_config in self.config.model.items():
                task = TuningTask(
                    board_name,
                    model_name,
                    board_config,
                    network_config,
                    self.tracker_port,
                    tune=self.config.tune,
                )
                self.worklist.append(task)
                self.tasks.append(task)


class BackendExperimentScheduler(ExperimentSchedulerBase):
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
                self.tracker_port,
                tune=False,
            )

            self.worklist.append(task)
            self.tasks.append(task)
