#
# Copyright (c) 2022 University of Tübingen.
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
"""Common functions for AOT model generation"""
import datetime
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
import tvm
from tvm import autotvm, relay
from tvm.contrib import graph_executor, utils
from tvm.micro import export_model_library_format
from tvm.micro.testing.utils import mlf_extract_workspace_size_bytes
from tvm.relay.backend import Executor, Runtime
from tvm.relay.backend.utils import mangle_module_name

_LOG = logging.getLogger(__name__)

NP_TYPE_TO_C = {
    "int8": "int8_t",
    "uint8": "uint8_t",
    "int16": "int16_t",
    "uint16": "uint16_t",
    "int32": "int32_t",
    "uint32": "uint32_t",
    "float32": "float",
}

AOT_SUCCESS_TOKEN = "AOT_TEST_SUCCESS"
AOT_FAILURE_TOKEN = "AOT_TEST_FAILURE"


class AOTModel(NamedTuple):
    """Class to describe a model under test

    Parameters
    ----------
    module: tvm.IRModule
        IRModule to generate AOT executor for
    inputs: Dict[str, np.array]
        Dict of input names to value arrays
    outputs: Dict[str, np.array]
        Dict of output names to value arrays
    output_tolerance: Optional[Union[int, float]]
        Allowed tolerance of the output
    name: str
        Name to use for this model
    params: Optional[Dict[str, np.array]]
        Dict of parameter names to value arrays
    extra_memory_in_bytes: int
        Extra memory to allocate after planned memory
    """

    module: tvm.IRModule
    inputs: Dict[str, np.array]
    outputs: Dict[str, np.array]
    output_tolerance: Optional[Union[int, float]] = None
    name: str = "default"
    params: Optional[Dict[str, np.array]] = None
    extra_memory_in_bytes: int = 0


class AOTCompiledModel(NamedTuple):
    """A compiled AOTTestModel with associated module

    Parameters
    ----------
    model: AOTTestModel
        Input model to be compiled
    module: tvm.runtime.Module
        The compiled Module for the associated AOTTestModel
    """

    model: AOTModel
    executor_factory: tvm.relay.backend.executor_factory.AOTExecutorFactoryModule


class AOTDataLinkage(NamedTuple):
    """A compiled AOTTestModel with associated module

    Parameters
    ----------
    section: str
        Named section to place data into
    alignment: int
        Section alignment
    """

    section: str
    alignment: int


def _mangle_name(mod_name, name):
    mod_name = mangle_module_name(mod_name)
    return mod_name + "_" + name


# TODO: Move to linker script with list of symbols rather than coding into source
def _emit_data_linkage(output_file, data_linkage):
    if data_linkage is not None:
        output_file.write(
            f'__attribute__((section("{data_linkage.section}"), '
            f"aligned({data_linkage.alignment}))) "
        )


def _emit_main_prologue(
    main_file,
    custom_prologue,
    workspace_bytes,
    data_linkage,
    compiled_models,
    interface_api,
    use_stack_allocator=True,
):
    if use_stack_allocator:
        workspace_define = f"#define WORKSPACE_SIZE ({workspace_bytes}"
        if interface_api == "c":
            for compiled_model in compiled_models:
                model = compiled_model.model
                workspace_define += f" + TVMGEN_{model.name.upper()}_WORKSPACE_SIZE"
        # Add TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES because of memory alignment.
        workspace_define += " + TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES)\n"
        main_file.write(workspace_define)
        _emit_data_linkage(main_file, data_linkage)
        main_file.write("static uint8_t g_aot_memory[WORKSPACE_SIZE];\n")
        main_file.write("tvm_workspace_t app_workspace;\n")
        main_file.write(
            """\n
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
    return StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
}
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
    return StackMemoryManager_Free(&app_workspace,ptr);
}
        """
        )
    else:
        # An implementation is not needed for these if the stack allocator is not used
        main_file.write(
            """\n
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
    return kTvmErrorFunctionCallNotImplemented;
}
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
    return kTvmErrorFunctionCallNotImplemented;
}
            """
        )
    #     main_file.write(
    #         """\n
    # void TVMPlatformAbort(tvm_crt_error_t code) { exit(-1); }
    # void TVMLogf(const char* msg, ...) {
    #   va_list args;
    #   va_start(args, msg);
    #   vfprintf(stdout, msg, args);
    #   va_end(args);
    # }\n
    # TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) {}
    main_file.write(
        """
int main(){\n
     """
    )
    main_file.write(custom_prologue)


def _emit_main_data(main_file, input_map, output_map, mod_name):
    for key in input_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(
            f'#include "{_mangle_name(mod_name,"input_data")}_{sanitized_tensor_name}.h"\n'
        )

    for key in output_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(
            f'#include "{_mangle_name(mod_name,"expected_output_data")}_'
            f'{sanitized_tensor_name}.h"\n'
            f'#include "{_mangle_name(mod_name,"output_data")}_'
            f'{sanitized_tensor_name}.h"\n'
        )


def _emit_main_device_structs(main_file, devices, mod_name):
    if devices:
        main_file.write(
            f"struct {_mangle_name(mod_name, 'devices')} {_mangle_name(mod_name, 'devices')} = {{"
        )
        for device in devices:
            main_file.write(f"\t.{device} = {device},\n")
        main_file.write("};\n")


def _emit_main_workspace_pool_structs(main_file, workspace_pool_names, mod_name):
    if workspace_pool_names and len(workspace_pool_names) > 0:
        main_file.write(
            f"struct {_mangle_name(mod_name, 'workspace_pools')} "
            f"{_mangle_name(mod_name, 'workspace_pools')} = {{"
        )
        for workspace_pool_name in workspace_pool_names.keys():
            main_file.write(
                f"\t.{workspace_pool_name} = {workspace_pool_names[workspace_pool_name]}"
                f"{workspace_pool_name},\n"
            )
        main_file.write("};\n")


def _emit_main_data_structs(main_file, input_map, output_map, mod_name):
    main_file.write(
        f"struct {_mangle_name(mod_name, 'inputs')} {_mangle_name(mod_name, 'inputs')} = {{"
    )
    for key in input_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(
            f"\t.{sanitized_tensor_name} = "
            f"{_mangle_name(mod_name, 'input_data')}_{sanitized_tensor_name},\n"
        )
    main_file.write("};\n")

    main_file.write(
        f"struct {_mangle_name(mod_name, 'outputs')} {_mangle_name(mod_name, 'outputs')} = {{"
    )
    for key in output_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(
            f"\t.{sanitized_tensor_name} = {_mangle_name(mod_name, 'output_data')}_"
            f"{sanitized_tensor_name},\n"
        )
    main_file.write("};\n")


def _emit_main_data_setup(main_file, input_map, output_map, mod_name):
    num_outputs = len(output_map)
    num_inputs = len(input_map)
    main_file.write(f'void* {_mangle_name(mod_name,"inputs")}[{num_inputs}] = {{ ')
    for key in input_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(
            f'{_mangle_name(mod_name,"input_data")}_{sanitized_tensor_name}, '
        )
    main_file.write("};\n")
    main_file.write(f'void* {_mangle_name(mod_name,"outputs")}[{num_outputs}]  = {{ ')
    for key in output_map:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        main_file.write(
            f'{_mangle_name(mod_name, "output_data")}_{sanitized_tensor_name}, '
        )
    main_file.write("};\n")


def _emit_main_c_interface_call(
    main_file, devices, workspace_pool_names, mod_name, use_workspace_io
):
    sub_strings = list()
    sub_strings.append(f'{_mangle_name(mod_name,"run")}(')
    if not use_workspace_io:
        sub_strings.append(f'&{_mangle_name(mod_name,"inputs")}, ')
        sub_strings.append(f'&{_mangle_name(mod_name,"outputs")}, ')
    if workspace_pool_names:
        sub_strings.append(f'&{_mangle_name(mod_name,"workspace_pools")}, ')
    if devices:
        sub_strings.append(f'&{_mangle_name(mod_name,"devices")}, ')
    # Removing the last two characters that is a comma and a space
    sub_strings[-1] = sub_strings[-1][:-2]
    # Adding brackets and newline instead
    sub_strings[-1] = sub_strings[-1] + ");\n"

    main_file_string = "".join(sub_strings)
    main_file.write(main_file_string)


def _emit_main_fake_packed_values(main_file):
    main_file.write(
        """
    static DLDevice fake_device = {kDLCPU, 0};
    static int64_t fake_dims = 0;
    static int64_t fake_shape = {0};
    """
    )


def _emit_main_packed_call(main_file, input_map, output_list, mod_name):
    tensors_name = _mangle_name(mod_name, "tensors")
    values_name = _mangle_name(mod_name, "values")
    typeids_name = _mangle_name(mod_name, "typeids")

    def fake_tensor(source, source_index, packed_index):
        main_file.write(
            f"""
        {tensors_name}[{packed_index}].device = fake_device;
        {tensors_name}[{packed_index}].data = {source}[{source_index}];
        {tensors_name}[{packed_index}].shape = &fake_shape;
        {tensors_name}[{packed_index}].ndim = fake_dims;
        {tensors_name}[{packed_index}].byte_offset = 0;
        {tensors_name}[{packed_index}].strides = NULL;
        {values_name}[{packed_index}].v_handle = &{tensors_name}[{packed_index}];
        """
        )

    num_outputs = len(output_list)
    num_inputs = len(input_map)
    num_tensors = num_inputs + num_outputs
    main_file.write(
        f"""
    DLTensor {tensors_name}[{num_tensors}];
    TVMValue {values_name}[{num_tensors}];
    int32_t {typeids_name}[{num_tensors}];
    """
    )

    for i in range(0, num_inputs):
        fake_tensor(_mangle_name(mod_name, "inputs"), i, i)
    for i in range(0, num_outputs):
        fake_tensor(_mangle_name(mod_name, "outputs"), i, i + num_inputs)

    main_file.write(
        f'{_mangle_name(mod_name, "run")}({values_name}, {typeids_name}, 0, NULL, 0, NULL);\n'
    )
    main_file.write("\n")


def _emit_main_compare(
    main_file, outputs, output_tolerance, mod_name, use_interface_c=False
):
    for key in outputs:
        sanitized_tensor_name = re.sub(r"\W", "_", key)
        expected_data_name = _mangle_name(
            mod_name, f"expected_output_data_{sanitized_tensor_name}"
        )
        is_float_dtype = outputs[key].dtype == "float32"

        comparison_function = "abs"
        tolerance = output_tolerance or 0
        if is_float_dtype:
            comparison_function = "fabs"
            tolerance = output_tolerance or 0.001

        data_length_var_name = (
            _mangle_name(mod_name, f"output_data_{sanitized_tensor_name}") + "_len"
        )
        if use_interface_c:
            c_type = NP_TYPE_TO_C[str(outputs[key].dtype)]
            actual_data_name = f"(({c_type}*)" + _mangle_name(
                mod_name, f"outputs.{sanitized_tensor_name})"
            )
        else:
            actual_data_name = _mangle_name(
                mod_name, f"output_data_{sanitized_tensor_name}"
            )
        main_file.write(
            f"for (int i = 0; i<{data_length_var_name}; i++) {{\n"
            f"\tif ({comparison_function}({actual_data_name}[i]-"
            f"{expected_data_name}[i]) > {tolerance}) {{\n"
            f'\t\tprintf("{AOT_FAILURE_TOKEN}\\n");\n'
            f"\t\treturn -1;\n"
            f"\t}}\n"
            f"}}"
        )


def _emit_main_init_memory_manager(main_file):
    main_file.write(
        "StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);"
    )
    main_file.write("\n")


def _emit_main_epilogue(main_file, custom_epilogue):
    main_file.write(custom_epilogue)
    main_file.write(f'printf("{AOT_SUCCESS_TOKEN}\\n");')
    main_file.write("return 0;")
    main_file.write("}\n")


def _emit_main_common_includes(main_file, custom_includes):
    main_file.write("#include <stdio.h>\n")
    main_file.write("#include <stdarg.h>\n")
    main_file.write("#include <stdlib.h>\n")
    main_file.write("#include <math.h>\n")
    main_file.write('#include "tvm/runtime/c_runtime_api.h"\n')
    main_file.write('#include "tvm/runtime/crt/stack_allocator.h"\n')
    for include in custom_includes:
        main_file.write(f'#include "{include}"\n')


def _emit_main_micro_include(main_file, mod_name):
    main_file.write(f"#include <{mangle_module_name(mod_name)}.h>\n")


def _create_main(
    test_name,
    compiled_models,
    output_path,
    custom_includes,
    custom_prologue,
    custom_epilogue,
    data_linkage,
    interface_api,
    workspace_bytes,
    use_stack_allocator=True,
    use_workspace_io=False,
):
    file_path = pathlib.Path(f"{output_path}/" + test_name).resolve()
    # create header file
    raw_path = file_path.with_suffix(".c").resolve()
    with open(raw_path, "w") as main_file:
        _emit_main_common_includes(main_file, custom_includes)
        if interface_api == "c":
            for compiled_model in compiled_models:
                model = compiled_model.model
                _emit_main_micro_include(main_file, model.name)

        for compiled_model in compiled_models:
            model = compiled_model.model
            _emit_main_data(main_file, model.inputs, model.outputs, model.name)

        _emit_main_prologue(
            main_file,
            custom_prologue,
            workspace_bytes,
            data_linkage,
            compiled_models,
            interface_api,
            use_stack_allocator,
        )
        if use_stack_allocator:
            _emit_main_init_memory_manager(main_file)

        if interface_api == "c":
            for compiled_model in compiled_models:
                model = compiled_model.model
                executor_codegen_metadata = (
                    compiled_model.executor_factory.executor_codegen_metadata
                )
                devices = compiled_model.executor_factory.get_devices()
                workspace_pool_names = {}
                if executor_codegen_metadata.pool_inputs:
                    workspace_pool_names = {
                        allocated_pool.pool_info.pool_name: "&"
                        if isinstance(
                            allocated_pool.pool_info,
                            tvm.ir.memory_pools.ConstantPoolInfo,
                        )
                        else ""
                        for allocated_pool in dict(
                            executor_codegen_metadata.pool_inputs
                        ).values()
                        if not allocated_pool.pool_info.is_internal
                    }
                _emit_main_device_structs(main_file, devices, model.name)
                if not use_workspace_io:
                    _emit_main_workspace_pool_structs(
                        main_file, workspace_pool_names, model.name
                    )
                    _emit_main_data_structs(
                        main_file, model.inputs, model.outputs, model.name
                    )
                _emit_main_c_interface_call(
                    main_file,
                    devices,
                    list(workspace_pool_names.keys()),
                    model.name,
                    use_workspace_io,
                )
        else:
            _emit_main_fake_packed_values(main_file)
            for compiled_model in compiled_models:
                model = compiled_model.model
                _emit_main_data_setup(
                    main_file, model.inputs, model.outputs, model.name
                )
                _emit_main_packed_call(
                    main_file, model.inputs, model.outputs, model.name
                )

        for compiled_model in compiled_models:
            model = compiled_model.model
            _emit_main_compare(
                main_file,
                model.outputs,
                model.output_tolerance,
                model.name,
                interface_api == "c",
            )
        _emit_main_epilogue(main_file, custom_epilogue)


def _create_header_file(tensor_name, npy_data, output_path, data_linkage):
    """
    This method generates a header file containing the data contained in the numpy array provided.
    It is used to capture the tensor data (for both inputs and expected outputs)
    to be bundled into the standalone application.
    """
    file_path = pathlib.Path(f"{output_path}/" + tensor_name).resolve()
    # create header file
    raw_path = file_path.with_suffix(".h").resolve()
    with open(raw_path, "w") as header_file:
        header_file.write("#include <stddef.h>\n")
        header_file.write("#include <stdint.h>\n")
        header_file.write("#include <dlpack/dlpack.h>\n")
        header_file.write(f"const size_t {tensor_name}_len = {npy_data.size};\n")

        _emit_data_linkage(header_file, data_linkage)

        header_file.write(f"{NP_TYPE_TO_C[str(npy_data.dtype)]} {tensor_name}[] =")

        header_file.write("{")
        for i in np.ndindex(npy_data.shape):
            header_file.write(f"{npy_data[i]}, ")
        header_file.write("};\n\n")


def build_aot_runner(
    compiled_models: List[AOTModel],
    prologue: str = "",
    epilogue: str = "",
    includes: List[str] = [],
    pass_config: Dict[str, Any] = {},
    interface_api: str = "c",
    debug_calculated_workspaces=False,
    workspace_byte_alignment=8,
    constant_byte_alignment=8,
    data_linkage: AOTDataLinkage = None,
    target_dir: str = None,
    use_workspace_io: bool = False,
    workspace_bytes: int = 256 * 1024,
):
    """This function generates a main function and associated data files to run models compiled for AOT runner

    Args:
        model (List[AOTModel]): _description_
        prologue (str): Code to prepend to the main function
        epilogue (str): Code to append to the main function
        includes (List[str]): Additional includes required to run the AOT test runner
        pass_config (Dict[str, Any]) Additional pass configuration when building the model
        debug_calculated_workspaces (bool, optional): _description_. Defaults to False.
        workspace_byte_alignment (int, optional): _description_. Defaults to 8.
        constant_byte_alignment (int, optional): _description_. Defaults to 8.
        data_linkage (AOTDataLinkage, optional): _description_. Defaults to None.
        target_dir (str, optional): _description_. Defaults to None.
        use_workspace_io (bool, optional): _description_. Defaults to False.
        workspace_bytes (int, optional): size of global workspace if USMP is not used. Defaults to 256 KB
    """

    def generate_body(base_path, workspace_bytes):
        base_path = os.path.abspath(base_path)
        build_path = base_path
        include_path = os.path.join(base_path, "include")

        os.makedirs(include_path, exist_ok=True)

        # Interface C APIs does not need compiler generated
        # workspace to generate the test application, because
        # workspace size is codegen'd as a macro to
        # tvmgen_<model_name>.h.

        workspace_bytes = 0 if interface_api == "c" else workspace_bytes

        for compiled_model in compiled_models:
            model = compiled_model.model
            workspace_bytes += model.extra_memory_in_bytes
            for key in model.inputs:
                print("Creating input:", key)
                sanitized_tensor_name = re.sub(r"\W", "_", key)
                _create_header_file(
                    f'{_mangle_name(model.name, "input_data")}_{sanitized_tensor_name}',
                    model.inputs[key],
                    include_path,
                    data_linkage,
                )

            for key in model.outputs:
                print("Creating output:", key)
                sanitized_tensor_name = re.sub(r"\W", "_", key)
                _create_header_file(
                    f'{_mangle_name(model.name, "output_data")}_{sanitized_tensor_name}',
                    np.zeros(model.outputs[key].shape, model.outputs[key].dtype),
                    include_path,
                    data_linkage,
                )
                _create_header_file(
                    f'{_mangle_name(model.name, "expected_output_data")}_{sanitized_tensor_name}',
                    model.outputs[key],
                    include_path,
                    data_linkage,
                )

        use_usmp = pass_config.get("tir.usmp.enable", False)
        # We only need the stack allocator if USMP is not used
        use_stack_allocator = not use_usmp

        _create_main(
            "test.c",
            compiled_models,
            build_path,
            includes,
            prologue,
            epilogue,
            data_linkage,
            interface_api,
            workspace_bytes,
            use_stack_allocator,
            use_workspace_io,
        )

    if target_dir is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_body(os.path.join(tmpdir, "test"), workspace_bytes)
    else:
        generate_body(target_dir, workspace_bytes)


def generate_ref_data(mod, input_data, params=None, target="llvm"):
    """Generate reference data through executing the relay module

    Args:
        mod (tvm.ir.Module): the module containing the main function for the neural network
        input_data (Dict[str, np.ndarray]): the input data for the neural network
        params (Dict[str, np.ndarray], optional): The parameters of of the neural network
        target (str, optional): tvm target to use for generation of reference data. Defaults to "llvm".

    Returns:
        Dict[str, np.ndarray]: Mapping of output tensor name to reference outputs
    """
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lib = relay.build(mod, target=target, params=params)

    lib_name = "mod.so"
    temp = utils.tempdir()
    lib_path = temp.relpath(lib_name)
    lib.export_library(lib_path)
    lib = tvm.runtime.load_module(lib_path)
    grt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    grt_mod.set_input(**input_data)
    grt_mod.run()
    output_count = grt_mod.get_num_outputs()
    out = [grt_mod.get_output(i).numpy() for i in range(output_count)]
    if isinstance(mod, tvm.relay.Function):
        main = mod
    else:
        main = mod["main"]
    if main.attrs is None or main.attrs["output_tensor_names"] is None:
        output_tensor_names = (
            ["output"]
            if output_count == 1
            else [f"output{i}" for i in range(output_count)]
        )
    else:
        output_tensor_names = main.attrs["output_tensor_names"]

    return dict(zip(output_tensor_names, out))
