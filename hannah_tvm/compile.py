import logging
import os  
import shutil

import hydra
import tvm
import tvm.relay as relay
import tvm.micro as micro
import tvm.autotvm as autotvm
from tvm.contrib import graph_runtime

from . import config 
from . import measure
from . import load

logger = logging.getLogger("hannah-tvm-compile")

def compile(config):
    relay_mod, params, inputs = load.load_model(config.model)
    #measure_context = measure.AutomateRPCMeasureContext(config.board)
    
    target = tvm.target.Target(config.board.target)
    target_host = None
    if config.board.target_host:
        target_host = tvm.target.Target(config.board.target_host)

    if target.kind == "cuda":
        if "arch" in target.attrs:
            autotvm.measure.measure_methods.set_cuda_target_arch(target.attrs["arch"])
        else:
            logger.warning("CUDA target has no architecture attribute")

    build_cfg = {}
    if target.kind == "c":
        build_cfg = { "tir.disable_vectorize": True }
 

    with tvm.transform.PassContext(opt_level=3, config=build_cfg):
         lib = relay.build(relay_mod, target=target, target_host=target_host, params=params)

    if config.board.micro:
        workspace = micro.Workspace(debug=True)
        opts = micro.default_options(os.path.join(micro.get_standalone_crt_dir(), "template", "host"))
        compiler = micro.DefaultCompiler(target=target)
        micro_binary = micro.build_static_runtime(
            workspace,
            compiler,
            lib.module,
            opts,
            extra_libs=[tvm.micro.get_standalone_crt_lib("memory")]
        )

        # Prepare target data
        outDir = "out"
        os.makedirs(outDir, exist_ok=True)
        shutil.copy2(workspace.path + "/src/module/lib1.c", outDir + "/kernels.c")
        shutil.copy2(workspace.path + "/src/module/lib0.c", outDir + "/syslib.c")
        with open(outDir + "/graph.json", "w") as f:
            f.write(lib.graph_json)
        with open(outDir + "/params.bin", "wb") as f:
            f.write(relay.save_param_dict(lib.params))
        
        #codegen.generateTargetCode(outDir + "/runtime_wrapper.c", lib.graph_json, relay.save_param_dict(lib.params), self.modelInfo)



@hydra.main(config_name="config", config_path="conf")
def main(config):
    return compile(config)


if __name__ == "__main__":
    main()