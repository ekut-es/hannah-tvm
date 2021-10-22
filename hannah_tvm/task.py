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
import tvm.contrib.debugger.debug_runtime
import numpy as np

from dataclasses import dataclass
from typing import Any
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from omegaconf import OmegaConf


from . import config
from . import measure
from . import load


logger = logging.getLogger(__name__)
manager = multiprocessing.Manager()


@dataclass
class ModelConfig:
    mod: Any
    params: Any
    inputs: Any


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
        self.results["status"] = "created"
        self.results["error"] = None

        self.database_file = (
            Path(__file__).parent.resolve()
            / ".."
            / "database"
            / board_key
            / "database.json"
        )

        name = f"tuning-task-{board_key}-{model_key}"
        super().__init__(name=name)

    def run(self):
        try:
            self.results["status"] = "running"
            if isinstance(self.model_config, ModelConfig):
                relay_mod, params, inputs = (
                    self.model_config.mod,
                    self.model_config.params,
                    self.model_config.inputs,
                )
            else:
                relay_mod, params, inputs = load.load_model(self.model_config)

            if self.board_config.desired_layouts:
                print(relay_mod)
                desired_layouts = self.board_config.desired_layouts
                if OmegaConf.is_config(desired_layouts):
                    desired_layouts = OmegaConf.to_container(desired_layouts)
                seq = tvm.transform.Sequential(
                    [
                        relay.transform.RemoveUnusedFunctions(),
                        relay.transform.ConvertLayout(desired_layouts),
                    ]
                )
                with tvm.transform.PassContext(opt_level=3):
                    relay_mod = seq(relay_mod)
                print(relay_mod)

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

            remote, rlib, lib = self._build_and_upload(relay_mod, params)
            self._evaluate(inputs, remote, rlib, lib)
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

            runner = auto_scheduler.RPCRunner(
                key=self.board_config.name, host="localhost", port=self.tracker_port
            )

            database_file = (
                None
            )  # database_file = str(self.database_file) if self.database_file.exists() else None
            logger.info("Loading database %s", str(database_file))
            logger.info("Begin tuning...")
            tuner = auto_scheduler.TaskScheduler(
                tasks, task_weights, load_log_file=database_file
            )
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=len(tasks) * 1024,
                builder="local",
                runner=runner,
                measure_callbacks=[auto_scheduler.RecordToFile(self.log_file)],
                verbose=1,
            )

            tuner.tune(tune_option, per_task_early_stopping=64, adapative_training=True)

            mode = "a+"
            if not self.database_file.exists():
                self.database_file.parent.mkdir(exist_ok=True, parents=True)
                mode = "w"

            if Path(self.log_file).exists():
                logger.info("Saving database: %s", str(self.database_file))
                with self.database_file.open(mode) as db:
                    with Path(self.log_file).open("r") as log:
                        db.write(log.read())

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
        return remote, rlib, lib

    def _evaluate(self, inputs, remote, rlib, lib):
        # Create graph executor
        logger.info("Start evaluation")
        if str(self.target.kind) == "cuda":
            dev = remote.cuda()
        else:
            dev = remote.cpu()
        module = tvm.contrib.graph_executor.GraphModule(rlib["default"](dev))
        logger.info("Set inputs")
        for name, val in inputs.items():
            data_tvm = tvm.nd.array(val)
            module.set_input(name, data_tvm)

        # Evaluate on Graph Executor
        logger.info("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, repeat=10, min_repeat_ms=500)
        prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        logger.info(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )

        self.results["latency"] = float(np.mean(prof_res))
        self.results["latency_std"] = float(np.std(prof_res))

        # Use debug Executor to get per operator runtime
        debug_module = tvm.contrib.debugger.debug_executor.GraphModuleDebug(
            rlib["debug_create"]("default", dev), [dev], lib.get_graph_json(), None
        )
        for name, val in inputs.items():
            data_tvm = tvm.nd.array(val)
            debug_module.set_input(name, data_tvm)
        debug_module.run()

    def __str__(self):
        s = f"TuningTask(board={self.board_key} model={self.model_key})"
        return s
