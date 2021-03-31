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

    if target.kind == "cuda":
        autotvm.measure.measure_methods.set_cuda_target_arch(target.attrs["arch"])


    # target = "llvm"
    # target_host = "llvm"
    # ctx = tvm.cpu(0)
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(relay_mod, target=target, target_host=target_host, params=params)

    #for batch in dataset:
    #    output = model(batch)
    #    traced_output = traced_model(batch)


@hydra.main(config_name="config", config_path="conf")
def main(config):
    return compile(config)


if __name__ == "__main__":
    main()