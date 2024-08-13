#
# Copyright (c) 2024 hannah-tvm contributors.
#
# This file is part of hannah-tvm.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah-tvm for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import datetime
import logging
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

from tvm.autotvm.measure.measure import MeasureErrorNo, MeasureResult, Runner

from .utils import populate_crt


class GVSOCRunner(Runner):
    id = 0

    clock_frequency = 50000000

    def __init__(self, template_dir) -> None:
        build_dir = Path("build")
        build_dir.mkdir(exist_ok=True)
        self.project_dir = build_dir.absolute() / f"autotvm_project_{GVSOCRunner.id}"
        GVSOCRunner.id += 1

        self.project_dir.mkdir()
        shutil.copy(template_dir / "Makefile", self.project_dir / "Makefile")
        shutil.copy(template_dir / "runner.h", self.project_dir / "runner.h")
        shutil.copy(template_dir / "test.c", self.project_dir / "test.c")
        shutil.copy(
            template_dir / "utvm_runtime_api.c", self.project_dir / "utvm_runtime_api.c"
        )
        shutil.copy(
            template_dir / "utvm_runtime_api.h", self.project_dir / "utvm_runtime_api.h"
        )

        populate_crt(self.project_dir)

        super().__init__()

    def get_build_kwargs(self):
        return {}

    def write_c_runner(self, arg_info):
        (self.project_dir / "build").mkdir(exist_ok=True)
        with open(self.project_dir / "build" / "runner.c", "w+") as f:
            f.write('#include "../runner.h"\n')
            f.write('#include "tvm/runtime/c_runtime_api.h"\n')
            f.write(
                "int default_function(TVMValue* args, int* type_codes, int num_args);\n"
            )

            for i, (shape, type) in enumerate(arg_info):
                match = re.match(r"(float|u?int)(\d+)", type)
                if match:
                    if match.group(1) == "float":
                        ctype = "float"
                        kDL = "kDLFloat"
                    else:
                        ctype = type + "_t"
                        kDL = "kDLUInt" if type[0] == "u" else "kDLInt"
                    dldatatype = f"{{ {kDL}, {match.group(2)}, 1 }}"
                else:
                    print("can not handle type: ", type)
                    exit(1)

                f.write(f"{ctype} data{i}{''.join(f'[{dim}]'for dim in shape)};\n")
                f.write(f"int64_t shape{i}[] = {{ {', '.join(map(str, shape))} }};\n")
                f.write(
                    f"DLTensor tensor{i} = {{ data{i}, {{ kDLCPU, 0 }}, {len(shape)}, {dldatatype}, shape{i}, NULL, 0 }};\n"
                )

            args = ", ".join(
                f"{{ .v_handle = &tensor{i} }}" for i in range(len(arg_info))
            )
            f.write(f"TVMValue args[] = {{ {args} }};\n")

            types = ", ".join(["kTVMDLTensorHandle"] * len(arg_info))
            f.write(f"int types[] = {{ {types} }};\n")

            f.write(
                f"int run(){{ return default_function(args, types, {len(arg_info)}); }}\n"
            )
            f.flush()
            f.close()

    def run(self, measure_inputs, build_results):
        results = []

        for result in build_results:
            if isinstance(result, MeasureResult):
                results.append(result)
                continue

            filename, arg_info, error, time_cost = result
            self.write_c_runner(arg_info)
            model_path = self.project_dir / "build" / "model"
            if model_path.exists():
                shutil.rmtree(model_path)
            model_path.mkdir()

            try:
                with tarfile.open(filename) as tar:
                    tar.extractall(model_path)
                build_result = subprocess.run(
                    ["make", "conf", "clean", "all"],
                    capture_output=True,
                    timeout=30,
                    cwd=self.project_dir,
                )

                timestamp = datetime.datetime.now().timestamp()

                if build_result.returncode != 0:
                    results.append(
                        MeasureResult(
                            (
                                "",
                                build_result.stderr.decode(),
                            ),
                            MeasureErrorNo.COMPILE_HOST,
                            0,
                            timestamp,
                        )
                    )
                    continue
            except subprocess.TimeoutExpired:
                results.append(
                    MeasureResult(
                        (
                            "",
                            "makefile timeout",
                        ),
                        MeasureErrorNo.COMPILE_HOST,
                        0,
                        timestamp,
                    )
                )
                continue

            try:
                output = subprocess.check_output(
                    ["make", "run", "-s"], timeout=30, cwd=self.project_dir
                )
                cycles = re.search(rb"cycles:(\d+)", output)
                if cycles:
                    ms = int(cycles.group(1)) / self.clock_frequency * 1000
                    results.append(
                        MeasureResult(ms, MeasureErrorNo.NO_ERROR, 0, timestamp)
                    )
                else:
                    results.append(
                        MeasureResult(
                            (
                                "",
                                "pulp runtime error",
                            ),
                            MeasureErrorNo.RUNTIME_DEVICE,
                            0,
                            timestamp,
                        )
                    )
            except subprocess.TimeoutExpired:
                results.append(
                    MeasureResult(
                        (
                            "",
                            "timeout",
                        ),
                        MeasureErrorNo.RUN_TIMEOUT,
                        0,
                        timestamp,
                    )
                )
            except subprocess.CalledProcessError as e:
                results.append(
                    MeasureResult(
                        (
                            "",
                            e.output.decode(),
                        ),
                        MeasureErrorNo.RUNTIME_DEVICE,
                        0,
                        timestamp,
                    )
                )
        return results
