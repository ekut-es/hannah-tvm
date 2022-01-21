import logging
import multiprocessing
import time
import traceback
import os

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

from omegaconf import OmegaConf


from . import config
from . import measure
from . import load


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    mod: Any
    params: Any
    inputs: Any


class TuningTask(multiprocessing.Process):
    def __init__(
        self,
        board_key,
        model_key,
        board_config,
        model_config,
        task_connector,
        tuner=None,
    ):
        self._task_connector = task_connector
        self.board_key = board_key
        self.model_key = model_key
        self.board_config = board_config
        self.model_config = model_config
        self.tuner = tuner
        self.log_file = f"{self.tuner}_{board_key}_{model_key}.json"

        self.results = {}

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
            self._task_connector.setup()
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
                desired_layouts = self.board_config.desired_layouts
                if OmegaConf.is_config(desired_layouts):
                    desired_layouts = OmegaConf.to_container(desired_layouts)
                seq = tvm.transform.Sequential(
                    [relay.transform.ConvertLayout(desired_layouts)]
                )
                with tvm.transform.PassContext(opt_level=3):
                    relay_mod = seq(relay_mod)

            if self.tuner == "auto_scheduler":
                start_time = time.time()
                self._run_autoscheduler(relay_mod, params)
                final_time = time.time()
                self.results["tuning_duration"] = final_time - start_time
            elif self.tuner == "autotvm":
                start_time = time.time()
                self._run_autotuner(relay_mod, params)
                final_time = time.time()
                self.results["tuning_duration"] = final_time - start_time
            elif self.tuner:
                raise Exception(f"Unknown tuner {self.tuner}")
            else:
                self.results["tuning_duration"] = 0.0

            lib = self._build(relay_mod, params)
            remote_handle = self._task_connector.upload(lib)
            self._evaluate(inputs, remote_handle)
            self.results["status"] = "finished"
        except Exception as e:
            logger.critical(
                "Tuning model %s on board %s failed", self.model_key, self.board_key
            )
            logger.critical(str(e))
            traceback.print_tb(e.__traceback__)

            self.results["status"] = "failed"
            self.results["error"] = e
        finally:
            self._task_connector.teardown()

    def _run_autotuner(self, relay_mod, params):
        logger.info("Running autotuner")

        early_stopping = 800

        builder = self._task_connector.builder("autotvm")
        runner = self._task_connector.runner("autotvm")

        measure_option = autotvm.measure_option(builder=builder, runner=runner)

        tasks = autotvm.task.extract_from_program(
            relay_mod["main"],
            target=self._task_connector.target(),
            params=params,
            ops=None,
        )

        for num, tsk in enumerate(tasks):
            prefix = f"Task {tsk.name} ({num+1}/{len(tasks)})"
            tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type="rank")

            tmp_log_file = self.log_file + ".tmp"
            if os.path.exists(tmp_log_file):
                os.remove(tmp_log_file)

            tsk_trial = min(1024, len(tsk.config_space))
            tuner_obj.tune(
                n_trial=tsk_trial,
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                    autotvm.callback.log_to_file(tmp_log_file),
                ],
            )

            autotvm.record.pick_best(tmp_log_file, self.log_file)
            os.remove(tmp_log_file)

    def _run_autoscheduler(self, relay_mod, params):

        hardware_params = self.board_config.get("hardware_params", None)
        if hardware_params is not None:
            hardware_params = auto_scheduler.HardwareParams(**hardware_params)

        logger.info("Extracting tasks ...")
        tasks, task_weights = auto_scheduler.extract_tasks(
            relay_mod["main"],
            params,
            self._task_connector.target(),
            hardware_params=hardware_params,
        )

        runner = self._task_connector.runner("auto_scheduler")
        builder = self._task_connector.builder("auto_scheduler")

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
            builder=builder,
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

    def _build(self, relay_mod, params):
        logger.info("Compile...")
        if self.tuner == "auto_scheduler":
            with auto_scheduler.ApplyHistoryBest(self.log_file):
                with tvm.transform.PassContext(
                    opt_level=3, config={"relay.backend.use_auto_scheduler": True}
                ):
                    lib = relay.build_module.build(
                        relay_mod, target=self._task_connector.target(), params=params
                    )
        elif self.tuner == "autotvm":
            if Path(self.log_file).exists():
                with autotvm.apply_history_best(self.log_file):
                    with tvm.transform.PassContext(opt_level=3):
                        lib = relay.build_module.build(
                            relay_mod,
                            target=self._task_connector.target(),
                            params=params,
                        )
            else:
                logger.warning("Could not find tuner logs in: %s", self.log_file)
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay.build_module.build(
                        relay_mod, target=self._task_connector.target(), params=params
                    )

        else:
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(
                    relay_mod, target=self._task_connector.target(), params=params
                )

        return lib

    def _evaluate(self, inputs, remote_handle):
        # Create graph executor
        logger.info("Start evaluation")

        prof_res = self._task_connector.measure(remote_handle, inputs)

        logger.info(
            "Mean inference time (std dev): %.2f us (%.2f us)"
            % (np.mean(prof_res), np.std(prof_res))
        )

        self.results["latency"] = float(np.mean(prof_res))
        self.results["latency_stdev"] = float(np.std(prof_res))

        debug_profile = self._task_connector.profile(remote_handle, inputs)

        if debug_profile is not None:
            logger.info("Profile information: %s", str(debug_profile))

    def __str__(self):
        s = f"TuningTask(board={self.board_key} model={self.model_key})"
        return s
