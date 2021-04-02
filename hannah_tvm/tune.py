import logging
import time
import hydra
import torch 
import tvm
import tvm.relay as relay
import tvm.auto_scheduler as auto_scheduler
import tvm.autotvm as autotvm
from tvm.contrib import graph_runtime

from . import config 
from . import measure
from . import load



def compile(config):
    relay_mod, params, inputs = load.load_model(config.model)
    measure_context = measure.AutomateRPCMeasureContext(config.board)
    
    target = tvm.target.Target(config.board.target)
    target_host = tvm.target.Target(config.board.target_host)

    autotvm.measure.measure_methods.set_cuda_target_arch(target.attrs["arch"])

    logging.info("Extracting tasks ...")
    tasks, task_weights = auto_scheduler.extract_tasks(relay_mod["main"], params, target, target_host)

    for idx, task in enumerate(tasks):
        logging.info("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        logging.info(task.compute_dag)


    runner = measure_context.runner
    

    logging.info("Begin tuning...")
    log_file = f"{config.board.name}.log"
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=20000,
        runner=runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)


@hydra.main(config_name="config")
def main(config):
    return compile(config)


if __name__ == "__main__":
    main()