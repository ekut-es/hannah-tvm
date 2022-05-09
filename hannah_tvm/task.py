import enum
import json
import logging
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tqdm
import tvm
import tvm.auto_scheduler as auto_scheduler
import tvm.autotvm as autotvm
import tvm.contrib.debugger.debug_runtime
import tvm.relay as relay
import tvm.rpc
import tvm.rpc.tracker
from matplotlib.style import available
from omegaconf import OmegaConf
from tvm.auto_scheduler import search_policy
from tvm.auto_scheduler.measure_record import dump_record_to_string

from hannah_tvm.dataset import PerformanceDataset
from hannah_tvm.tuner.autotvm.callbacks import (
    progress_callback as autotvm_progress_callback,
)

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
        tuner=None,
    ):
        self._task_connector = task_connector
        self.board_key = board_key
        self.model_key = model_key
        self.board_config = board_config
        self.model_config = model_config
        self.tuner_config = tuner
        tuner_name = self.tuner_config.name if self.tuner_config else "baseline"
        self.tuner_log_file = f"{board_key}_{model_key}_{tuner_name}.json"

        self.results = {}

        self.results["board"] = board_key
        self.results["model"] = model_key
        self.results["error"] = None

        self.name = f"tuning-task-{board_key}-{model_key}"
        self.dataset: Optional[PerformanceDataset] = None

        self.status = TaskStatus.CREATED

    def run(self) -> None:
        try:
            self._task_connector.setup()

            target = self._task_connector.target()
            self.dataset = PerformanceDataset(self.board_config.name, target.kind)

            self.status = TaskStatus.RUNNING
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
                    [
                        relay.transform.InferType(),
                        relay.transform.DynamicToStatic(),
                        relay.transform.ConvertLayout(desired_layouts),
                        relay.transform.InferType(),
                    ]
                )
                with tvm.transform.PassContext(opt_level=3):
                    relay_mod = seq(relay_mod)

            self.dataset.add_program(self.model_key, relay_mod, params)

            logger.info("Starting tuning with config:")
            for k, v in self.tuner_config.items():
                logger.info("  %s, %s", str(k), str(v))

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
            elif self.tuner_config.name == "baseline":
                self.results["tuning_duration"] = 0.0
            else:
                raise Exception(f"Unknown tuner {self.tuner_config.name}")

            lib = self._build(relay_mod, params)
            remote_handle = self._task_connector.upload(lib)
            self._evaluate(inputs, remote_handle)
            self.status = TaskStatus.FINISHED

        except Exception as e:
            logger.critical(
                "Tuning model %s on board %s failed", self.model_key, self.board_key
            )
            logger.critical(str(e))
            traceback.print_tb(e.__traceback__)

            self.status = TaskStatus.FAILED
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

        self.dataset.add_tasks("autotvm", self.model_key, tasks, None)

        logger.info("Extracted %d tasks", len(tasks))

        pretrained_results = self.dataset.load_tuning_results("autotvm", tasks)
        logger.info("Loaded %d pretrained tuning results", len(pretrained_results))

        for num, tsk in enumerate(tasks):
            prefix = f"Task {tsk.name} ({num+1}/{len(tasks)})"
            if self.tuner_config.mode == "xgb":
                tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type="reg")
            elif self.tuner_config.mode == "xgb_rank":
                tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type="rank")
            elif self.tuner_config.mode == "random":
                tuner_obj = autotvm.tuner.RandomTuner(tsk)
            else:
                raise Exception(
                    "Tuner mode: %s is unknown for autotvm", self.tuner_config.mode
                )

            tuner_obj.load_history(pretrained_results)

            tmp_log_file = self.tuner_log_file + ".tmp"
            if os.path.exists(tmp_log_file):
                os.remove(tmp_log_file)

            tsk_trial = min(self.tuner_config.task_budget, len(tsk.config_space))

            tuner_obj.tune(
                n_trial=tsk_trial,
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.log_to_file(tmp_log_file),
                    autotvm.callback.progress_bar(tsk_trial),
                ],
            )

            lines = list(open(tmp_log_file).readlines())
            records = [
                rec for rec in map(autotvm.record.decode, lines) if rec is not None
            ]

            self.dataset.add_tuning_results("autotvm", records)

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

        self.dataset.add_tasks("auto_scheduler", self.model_key, tasks, task_weights)
        available_measurements = self.dataset.load_tuning_results(
            "auto_scheduler", tasks
        )

        preloaded_measurements = 0
        with open(self.tuner_log_file, "w") as log_f:
            for inp, res in available_measurements:
                log_f.write(dump_record_to_string(inp, res))
                preloaded_measurements += 1

        logger.info("Preloaded %d measurements", preloaded_measurements)

        runner = self._task_connector.runner("auto_scheduler")
        builder = self._task_connector.builder("auto_scheduler")

        if self.tuner_config.mode == "xgb":
            search_policy = "sketch.xgb"
        elif self.tuner_config.mode == "random":
            search_policy = "sketch.random"
        try:
            if self.tuner_config.equal_task_budget:
                for num, task in enumerate(tasks):
                    tuner = auto_scheduler.TaskScheduler(
                        [task], task_weights=None, load_log_file=self.tuner_log_file
                    )

                    tune_option = auto_scheduler.TuningOptions(
                        num_measure_trials=self.tuner_config.task_budget,
                        builder=builder,
                        runner=runner,
                        measure_callbacks=[
                            auto_scheduler.RecordToFile(self.tuner_log_file)
                        ],
                        verbose=1,
                    )
                    tuner.tune(
                        tune_option,
                        per_task_early_stopping=64,
                        adapative_training=True,
                        search_policy=search_policy,
                    )
            else:
                tuner = auto_scheduler.TaskScheduler(
                    tasks, task_weights=task_weights, load_log_file=self.tuner_log_file
                )
                tune_option = auto_scheduler.TuningOptions(
                    num_measure_trials=self.tuner_config.task_budget * len(tasks),
                    builder=builder,
                    runner=runner,
                    measure_callbacks=[
                        auto_scheduler.RecordToFile(self.tuner_log_file)
                    ],
                    verbose=1,
                )
                tuner.tune(
                    tune_option,
                    per_task_early_stopping=64,
                    adapative_training=True,
                    search_policy=search_policy,
                )
        finally:
            record_reader = auto_scheduler.RecordReader(self.tuner_log_file)
            records = record_reader.read_lines(skip_lines=preloaded_measurements)
            records = zip(*records)
            self.dataset.add_tuning_results("auto_scheduler", records)

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

        result = {}
        result["Duration (us)"] = prof_res.tolist()
        if debug_profile is not None:
            logger.info("Profile information: %s", str(debug_profile))
            json_profile = debug_profile.json()
            dict_profile = json.loads(json_profile)
            result.update(dict_profile)
        self.dataset.add_measurement(self.model_key, dict_profile)

    def __str__(self):
        s = f"TuningTask(board={self.board_key} model={self.model_key})"
        return s
