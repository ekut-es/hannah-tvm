import logging
import multiprocessing as mp
import os
import time
import traceback
import enum
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tvm
import tvm.auto_scheduler as auto_scheduler
import tvm.autotvm as autotvm
import tvm.contrib.debugger.debug_runtime
import tvm.relay as relay
import tvm.rpc
import tvm.rpc.tracker
from hannah_tvm.dataset import PerformanceDataset
from hannah_tvm.tuner.autotvm.callbacks import (
    progress_callback as autotvm_progress_callback,
)
from omegaconf import OmegaConf

from . import config, load

logger = logging.getLogger(__name__)


class TaskStatus(enum.IntEnum):
    CREATED = 1
    RUNNING = 2
    FINISHED = 3
    FAILED = 4

    @property
    def display_name(self):
        return self.name.lower()


@dataclass
class ModelConfig:
    mod: Any
    params: Any
    inputs: Any


class TuningTask:
    """Represents a full network tuning task"""

    def __init__(
        self,
        board_key,
        model_key,
        board_config,
        model_config,
        task_connector,
        status: mp.Value,
        progress: mp.Value,
        tuner=None,
    ):
        self._task_connector = task_connector
        self.board_key = board_key
        self.model_key = model_key
        self.board_config = board_config
        self.model_config = model_config
        self.tuner_config = tuner
        self.tuner_log_file = f"{board_key}_{model_key}.json"

        self.results = {}

        self.results["board"] = board_key
        self.results["model"] = model_key
        self.results["error"] = None

        self.database_file = (
            Path(__file__).parent.resolve()
            / ".."
            / "database"
            / board_key
            / "database.json"
        )

        name = f"tuning-task-{board_key}-{model_key}"
        # Handle to child Process running this task if task is run in a different process
        self.process = None
        self.dataset = None

        # Current status implemented as multiprocessing.value for sharing between running tasks
        self.status = status
        self.progress = progress

    def run(self, lock: mp.Lock = None, status: mp.Value = None) -> None:
        if status is not None:
            self.status = status
        try:
            self._task_connector.setup()

            target = self._task_connector.target()
            self.dataset = PerformanceDataset(self.board_config.name, target.kind, lock)

            self.status.value = TaskStatus.RUNNING.value
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

            self.dataset.add_program(self.model_key, relay_mod)

            if self.tuner_config.name == "auto_scheduler":
                start_time = time.time()
                self._run_autoscheduler(relay_mod, params)
                final_time = time.time()
                self.results["tuning_duration"] = final_time - start_time
            elif self.tuner_config.name == "autotvm":
                start_time = time.time()
                self._run_autotuner(relay_mod, params)
                final_time = time.time()
                self.results["tuning_duration"] = final_time - start_time
            elif self.tuner_config.name:
                raise Exception(f"Unknown tuner {self.tuner_config.name}")
            else:
                self.results["tuning_duration"] = 0.0

            lib = self._build(relay_mod, params)
            remote_handle = self._task_connector.upload(lib)
            self._evaluate(inputs, remote_handle)
            self.status.value = TaskStatus.FINISHED.value

        except Exception as e:
            logger.critical(
                "Tuning model %s on board %s failed", self.model_key, self.board_key
            )
            logger.critical(str(e))
            traceback.print_tb(e.__traceback__)

            self.status.value = TaskStatus.FAILED.value
            self.results["error"] = e
        finally:
            self._task_connector.teardown()

    def _run_autotuner(self, relay_mod, params):
        logger.info("Running autotuner")

        early_stopping = 800

        builder = self._task_connector.builder("autotvm")
        runner = self._task_connector.runner("autotvm")

        measure_option = autotvm.measure_option(builder=builder, runner=runner)

        logger.info("Extracting tuning tasks")
        tasks = autotvm.task.extract_from_program(
            relay_mod["main"],
            target=self._task_connector.target(),
            params=params,
            ops=None,
        )

        logger.info("Extracted %d tasks", len(tasks))

        for num, tsk in enumerate(tasks):
            self.progress.value = num / len(tasks)
            prefix = f"Task {tsk.name} ({num+1}/{len(tasks)})"
            tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type="rank")

            tmp_log_file = self.tuner_log_file + ".tmp"
            if os.path.exists(tmp_log_file):
                os.remove(tmp_log_file)

            tsk_trial = min(1024, len(tsk.config_space))
            step_progress = 1 / (len(tasks) * tsk_trial)

            tuner_obj.tune(
                n_trial=tsk_trial,
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm_progress_callback(step_progress, self.progress),
                    autotvm.callback.log_to_file(tmp_log_file),
                    autotvm.callback.progress_bar(tsk_trial),
                ],
            )

            autotvm.record.pick_best(tmp_log_file, self.tuner_log_file)
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

        database_file = str(self.database_file) if self.database_file.exists() else None

        logger.info("Loading database %s", str(database_file))
        logger.info("Begin tuning...")
        tuner = auto_scheduler.TaskScheduler(
            tasks, task_weights, load_log_file=database_file
        )
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=len(tasks) * 1024,
            builder=builder,
            runner=runner,
            measure_callbacks=[auto_scheduler.RecordToFile(self.tuner_log_file)],
            verbose=1,
        )

        tuner.tune(tune_option, per_task_early_stopping=64, adapative_training=True)

        mode = "a+"
        if not self.database_file.exists():
            self.database_file.parent.mkdir(exist_ok=True, parents=True)
            mode = "w"

        if Path(self.tuner_log_file).exists():
            logger.info("Saving database: %s", str(self.database_file))
            with self.database_file.open(mode) as db:
                with Path(self.tuner_log_file).open("r") as log:
                    db.write(log.read())

    def _build(self, relay_mod, params):
        logger.info("Compile...")
        if self.tuner_config.name == "auto_scheduler":
            with auto_scheduler.ApplyHistoryBest(self.tuner_log_file):
                with tvm.transform.PassContext(
                    opt_level=3, config={"relay.backend.use_auto_scheduler": True}
                ):
                    lib = relay.build_module.build(
                        relay_mod, target=self._task_connector.target(), params=params
                    )
        elif self.tuner_config.name == "autotvm":
            if Path(self.tuner_log_file).exists():
                with autotvm.apply_history_best(self.tuner_log_file):
                    with tvm.transform.PassContext(opt_level=3):
                        lib = relay.build_module.build(
                            relay_mod,
                            target=self._task_connector.target(),
                            params=params,
                        )
            else:
                logger.warning("Could not find tuner logs in: %s", self.tuner_log_file)
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

    def is_alive(self):
        if self.process:
            return self.process.is_alive()

        return False
