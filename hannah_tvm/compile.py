import logging
import os
from pathlib import Path
import shutil
import tempfile
import tarfile

import hydra
import tvm
from tvm.micro import model_library_format
import tvm.relay as relay
import tvm.micro as micro
import tvm.autotvm as autotvm
from tvm.contrib import graph_runtime

from . import config
from . import measure
from . import load

logger = logging.getLogger("hannah-tvm-compile")


def compile(config):
    for board_name, board in config.board.items():
        for model_name, model in config.model.items():
            logger.info("Compiling model %s for board %s", model_name, board_name)
            relay_mod, params, inputs = load.load_model(model)

            target = tvm.target.Target(board.target, host=board.target_host)

            build_cfg = {}
            if str(target.kind) == "c":
                build_cfg = {"tir.disable_vectorize": True}

            from . import pass_instrument

            with tvm.transform.PassContext(
                opt_level=3,
                config=build_cfg,
                instruments=[pass_instrument.PrintIR("all")],
            ):
                module = relay.build(relay_mod, target=target, params=params)
                if board.micro:
                    target_aot = tvm.target.Target(
                        board.target + " -link-params=1--executor=aot"
                    )
                    module_aot = relay.build(
                        relay_mod, target=target_aot, params=params
                    )

            if board.micro:
                logger.info("Building micro target")

                model_library_format_tar_path = Path("model_host_driven.tar")
                model_library_format_aot_path = Path("model_aot.tar")
                if model_library_format_tar_path.exists():
                    os.unlink(model_library_format_tar_path)

                tvm.micro.export_model_library_format(
                    module, model_library_format_tar_path
                )

                tvm.micro.export_model_library_format(
                    module_aot, model_library_format_aot_path
                )

                with tarfile.open(model_library_format_tar_path, "r:*") as tar_f:
                    print("\n".join(f" - {m.name}" for m in tar_f.getmembers()))

                build_dir = Path("build")
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                build_dir.mkdir()
                logger.info("Building in directory: %s", build_dir.absolute())
                generated_project_dir = build_dir.absolute() / "generated-project"
                generated_project = tvm.micro.generate_project(
                    board.micro.template_dir,
                    module,
                    generated_project_dir,
                    dict(board.micro.project_options),
                )

                # Build and flash the project
                generated_project.build()
                generated_project.flash()


@hydra.main(config_name="config", config_path="conf")
def main(config):
    return compile(config)


if __name__ == "__main__":
    main()
