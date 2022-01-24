
from pathlib import Path
import shutil
import subprocess
import re
import tarfile

from tvm.autotvm.measure.measure import Runner, MeasureResult, MeasureErrorNo

class PulpRunner(Runner):
    id = 0
    def __init__(self, template_dir) -> None:
        build_dir = Path("build")
        build_dir.mkdir(exist_ok=True)
        self.project_dir = build_dir.absolute() / f"autotvm_project_{PulpRunner.id}"
        PulpRunner.id += 1

        self.project_dir.mkdir()
        shutil.copy(template_dir / "Makefile", self.project_dir / "Makefile")
        shutil.copy(template_dir / "runner.h", self.project_dir / "runner.h")
        shutil.copy(template_dir / "test.c", self.project_dir / "test.c")
        shutil.copy(template_dir / "utvm_runtime_api.c", self.project_dir / "utvm_runtime_api.c")
        shutil.copy(template_dir / "utvm_runtime_api.h", self.project_dir / "utvm_runtime_api.h")

        super().__init__()

    def get_build_kwargs(self):
        return {}

    def write_c_runner(self, arg_info):
        (self.project_dir / "build").mkdir(exist_ok=True)
        with open(self.project_dir / "build" / "runner.c", "w+") as f:
            f.write("#include \"../runner.h\"\n")
            f.write("#include \"tvm/runtime/c_runtime_api.h\"\n")
            f.write("int default_function(TVMValue* args, int* type_codes, int num_args);\n")
            for i, (shape, type) in enumerate(arg_info):
                if type == "float32":
                    ctype = "float"
                    dldatatype = "{ kDLFloat, 32, 1 }"
                elif type == "int32":
                    ctype = "int32_t"
                    dldatatype = "{ kDLInt, 32, 1 }"
                elif type == "int16":
                    ctype = "int16_t"
                    dldatatype = "{ kDLInt, 16, 1 }"
                elif type == "int8":
                    ctype = "int8_t"
                    dldatatype = "{ kDLInt, 8, 1 }"
                elif type == "uint32":
                    ctype = "uint32_t"
                    dldatatype = "{ kDLUInt, 32, 1 }"
                elif type == "uint16":
                    ctype = "uint16_t"
                    dldatatype = "{ kDLUInt, 16, 1 }"
                elif type == "uint8":
                    ctype = "uint8_t"
                    dldatatype = "{ kDLUInt, 8, 1 }"
                else:
                    print("can not handle type: ", type)
                    exit(1)

                f.write("%s data%i%s;\n" % (ctype, i, "".join("[" + str(dim) + "]" for dim in shape)))
                f.write("int64_t shape%i[] = { %s };\n" % (i, ", ".join(str(dim) for dim in shape)))
                f.write("DLTensor tensor%i = { data%i, { kDLCPU, 0 }, %i, %s, shape%i, NULL, 0 };\n"
                        % (i, i, len(shape), dldatatype, i))
            f.write("TVMValue args[] = { %s };\n" % ", ".join("{ .v_handle = &tensor%i }" % i for i in range(len(arg_info))))
            f.write("int types[] = { %s };\n" % ((", kTVMDLTensorHandle" * (len(arg_info)))[2:] ))
            f.write("int run(){return default_function(args, types, %i);}\n" % len(arg_info))
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
                build_result = subprocess.run(["make", "conf", "clean", "all"], capture_output=True, timeout=30, cwd=self.project_dir)
                if build_result.returncode != 0:
                    results.append(MeasureResult(
                        (build_result.stderr.decode(),),
                        MeasureErrorNo.COMPILE_HOST,
                        0,
                        0
                    ))
                    continue
            except subprocess.TimeoutExpired:
                results.append(MeasureResult(
                    ("makefile timeout",),
                    MeasureErrorNo.COMPILE_HOST,
                    0,
                    0
                ))
                continue

            try:
                output = subprocess.check_output(["make", "run", "-s"], timeout=30, cwd=self.project_dir)
                cycles = re.search(rb"cycles:(\d+)", output)
                if cycles:
                    results.append(MeasureResult(
                        int(cycles.group(1)),
                        MeasureErrorNo.NO_ERROR,
                        0,
                        0
                    ))
                else:
                    results.append(MeasureResult(
                        ("pulp runtime error",),
                        MeasureErrorNo.RUNTIME_DEVICE,
                        0,
                        0
                    ))
            except subprocess.TimeoutExpired:
                results.append(MeasureResult(
                    ("timeout",),
                    MeasureErrorNo.RUN_TIMEOUT,
                    0,
                    0
                ))
            except subprocess.CalledProcessError as e:
                results.append(MeasureResult(
                    (e.output.decode(),),
                    MeasureErrorNo.RUNTIME_DEVICE,
                    0,
                    0
                ))
        return results
