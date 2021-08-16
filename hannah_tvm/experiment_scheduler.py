import logging
import multiprocessing
import time
import traceback

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


logger = logging.getLogger(__name__)
manager = multiprocessing.Manager()


class TuningTask(multiprocessing.Process):
    def __init__(
        self, board_key, model_key, board_config, model_config, tracker_port, tune=False
    ):
        self.board_key = board_key
        self.model_key = model_key
        self.board_config = board_config
        self.model_config = model_config
        self.tracker_port = tracker_port
        self.tune = tune
        self.log_file = f"auto_scheduler_{board_key}_{model_key}.json"

        self.target = tvm.target.Target(self.board_config.target)
        self.target_host = tvm.target.Target(self.board_config.target_host)
        self.results = manager.dict()

        self.results["board"] = board_key
        self.results["model"] = model_key
        # self.results["tasks"] = manager.dict()
        self.results["status"] = "created"
        self.results["error"] = None

        name = f"tuning-task-{board_key}-{model_key}"
        super().__init__(name=name)

    def run(self):
        try:
            self.results["status"] = "running"
            relay_mod, params, inputs = load.load_model(self.model_config)

            if str(self.target.kind) == "cuda":
                if "arch" in self.target.attrs:
                    logger.info(
                        "Setting cuda target arch %s", self.target.attrs["arch"]
                    )
                    autotvm.measure.measure_methods.set_cuda_target_arch(
                        self.target.attrs["arch"]
                    )
                else:
                    logger.warning("CUDA target has no architecture attribute")

            start_time = time.time()
            self._run_autoscheduler(relay_mod, params)
            final_time = time.time()
            self.results["tuning_duration"] = final_time - start_time

            remote, rlib = self._build_and_upload(relay_mod, params)
            self._evaluate(inputs, remote, rlib)
            self.results["status"] = "finished"
        except Exception as e:
            logger.critical(
                "Tuning model %s on board %s failed", self.model_key, self.board_key
            )
            logger.critical(str(e))
            traceback.print_tb(e.__traceback__)

            self.results["status"] = "failed"
            self.results["error"] = e

    def _run_autoscheduler(self, relay_mod, params):

        hardware_params = self.board_config.get("hardware_params", None)
        if hardware_params:
            hardware_params = auto_scheduler.HardwareParams(**hardware_params)

        if self.tune:
            logger.info("Extracting tasks ...")
            tasks, task_weights = auto_scheduler.extract_tasks(
                relay_mod["main"],
                params,
                self.target,
                self.target_host,
                hardware_params=hardware_params,
            )

            for idx, task in enumerate(tasks):
                logger.info(
                    "========== Task %d  (workload key: %s) =========="
                    % (idx, task.workload_key)
                )

            runner = auto_scheduler.RPCRunner(
                key=self.board_config.name, host="localhost", port=self.tracker_port
            )

            logger.info("Begin tuning...")
            tuner = auto_scheduler.TaskScheduler(
                tasks, task_weights, callbacks=[measure.PrintPBarInfo(self.results)]
            )
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=len(tasks) * 512,
                builder="local",
                runner=runner,
                measure_callbacks=[auto_scheduler.RecordToFile(self.log_file)],
                verbose=0,
            )

            tuner.tune(tune_option)

    def _build_and_upload(self, relay_mod, params):
        logger.info("Compile...")
        if self.tune:
            with auto_scheduler.ApplyHistoryBest(self.log_file):
                with tvm.transform.PassContext(
                    opt_level=3, config={"relay.backend.use_auto_scheduler": True}
                ):
                    lib = relay.build_module.build(
                        relay_mod,
                        target=self.target,
                        target_host=self.target_host,
                        params=params,
                    )
        else:
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(
                    relay_mod,
                    target=self.target,
                    target_host=self.target_host,
                    params=params,
                )

        # Export library
        tmp = tvm.contrib.utils.tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # Upload module to device
        logger.info("Upload...")
        remote = auto_scheduler.utils.request_remote(
            self.board_config.name, "localhost", self.tracker_port, timeout=10000
        )

        remote.upload(tmp.relpath(filename))

        rlib = remote.load_module(filename)
        logger.info("Upload finished")
        return remote, rlib

    def _evaluate(self, inputs, remote, rlib):
        # Create graph executor
        logger.info("Start evaluation")
        print(remote)
        dev = remote.cpu()
        module = tvm.contrib.graph_executor.GraphModule(rlib["default"](dev))
        logger.info("Set inputs")
        for name, val in inputs.items():
            logger.info("  %s", name)
            data_tvm = tvm.nd.array(val)
            module.set_input(name, data_tvm)

        # Evaluate
        logger.info("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, repeat=10, min_repeat_ms=500)
        prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        logger.info(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )

    def __str__(self):
        s = f"TuningTask(board={self.board_key} model={self.model_key})"
        return s


class ExperimentScheduler:
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

    def run(self):
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

        with tqdm(total=len(self.worklist)) as pbar:
            with logging_redirect_tqdm():
                while self.worklist or self.running_tasks:
                    self.report()
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
                                task = self.runing_tasks[board_name]
                                self._start_server(task.board_config)

                    board_summary = self.tracker_conn.summary()

                    if len(self.running_tasks) >= self.n_jobs:
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
                                        task.start()
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
