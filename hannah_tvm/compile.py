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
from .compiler import Compiler_Ext, get_compiler_options

logger = logging.getLogger("hannah-tvm-compile")


def compile(config):
    for board_name, board in config.board.items():
        for model_name, model in config.model.items():
            logger.info("Compiling model %s for board %s", model_name, board_name)
            relay_mod, params, inputs = load.load_model(model)

            target = tvm.target.Target(board.target)
            target_host = target
            if board.target_host:
                target_host = tvm.target.Target(board.target_host)

            if target.kind == "cuda":
                if "arch" in target.attrs:
                    autotvm.measure.measure_methods.set_cuda_target_arch(
                        target.attrs["arch"]
                    )
                else:
                    logger.warning("CUDA target has no architecture attribute")

            build_cfg = {}
            if str(target.kind) == "c":
                build_cfg = {"tir.disable_vectorize": True}

            print(build_cfg)
            print(relay_mod)

            with tvm.transform.PassContext(opt_level=3, config=build_cfg):
                lib = relay.build(relay_mod, target=target, params=params)

            if board.micro:
                logger.info("Building micro target")

                for i, m in enumerate(lib.module._collect_dso_modules()):
                    with open("/tmp/build/lib" + str(i) + ".ll", "w") as file:
                        file.write(m.get_source())
                workspace = micro.Workspace(debug=True)
                opts = get_compiler_options(board.micro)
                compiler = hydra.utils.instantiate(board.micro.compiler)

                micro_binary = micro.build_static_runtime(
                    workspace,
                    compiler,
                    lib.module,
                    opts,
                    extra_libs=[tvm.micro.get_standalone_crt_lib("memory")],
                )

                # Prepare target data
                outDir = "out"
                os.makedirs(outDir, exist_ok=True)
                shutil.copy2(
                    workspace.path + "/src/module/lib1.c", outDir + "/kernels.c"
                )
                shutil.copy2(
                    workspace.path + "/src/module/lib0.c", outDir + "/syslib.c"
                )
                with open(outDir + "/graph.json", "w") as f:
                    f.write(lib.graph_json)
                with open(outDir + "/params.bin", "wb") as f:
                    f.write(relay.save_param_dict(lib.params))

            # codegen.generateTargetCode(outDir + "/runtime_wrapper.c", lib.graph_json, relay.save_param_dict(lib.params), self.modelInfo)


@hydra.main(config_name="config", config_path="conf")
def main(config):
    return compile(config)


if __name__ == "__main__":
    main()
