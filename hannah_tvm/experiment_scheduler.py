import logging
import tvm
import tvm.auto_scheduler as auto_scheduler
import tvm.autotvm as autotvm

from . import config
from . import measure
from . import load


logger = logging.getLogger(__name__)


class TuningTask:
    def __init__(self, board_config, model_config):
        self.board_config = board_config
        self.model_config = model_config

    def run(self):
        self._run_autoscheduler()

    def _run_autoscheduler(self):
        relay_mod, params, inputs = load.load_model(config.model)
        with measure.AutomateRPCMeasureContext(config.board) as measure_context:

            target = tvm.target.Target(config.board.target)
            target_host = tvm.target.Target(config.board.target_host)

            if str(target.kind) == "cuda":
                if "arch" in target.attrs:
                    logging.info("Setting cuda target arch %s", target.attrs["arch"])
                    autotvm.measure.measure_methods.set_cuda_target_arch(
                        target.attrs["arch"]
                    )
                else:
                    logger.warning("CUDA target has no architecture attribute")

            logger.info("Extracting tasks ...")
            hardware_params = config.board.get("hardware_params", None)
            if hardware_params:
                hardware_params = auto_scheduler.HardwareParams(**hardware_params)

            tasks, task_weights = auto_scheduler.extract_tasks(
                relay_mod["main"],
                params,
                target,
                target_host,
                hardware_params=hardware_params,
            )

            for idx, task in enumerate(tasks):
                logger.info(
                    "========== Task %d  (workload key: %s) =========="
                    % (idx, task.workload_key)
                )
                logger.info(task.compute_dag)

            runner = measure_context.runner

            logger.info("Begin tuning...")
            log_file = f"{config.board.name}.log"
            tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=20000,
                builder="local",
                runner=runner,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            )

            tuner.tune(tune_option)

    def __str__(self):
        s = "TuningTask:\n"
        s += str(self.board_config)
        s += "\n"
        s += str(self.model_config)
        s += "\n"

        return s


class ExperimentScheduler:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        for board_name, board_config in self.config.board.items():
            for model_name, network_config in self.config.model.items():
                task = TuningTask(board_config, network_config)
                print(task)
