#
# Copyright (c) 2023 University of Tübingen.
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
import json
import re
from math import prod

import tvm
from tvm import relay


class GVSOCBuilder:
    # Generate c code to execute the given json graph
    def build(self, graph, params, file):
        # include headers
        file.write('#include "../runner.h"\n#include "tvm/runtime/c_runtime_api.h"\n\n')

        # declare lookup function for linked parameters
        file.write(
            "int _lookup_linked_param(TVMValue *args, int *type_codes, int num_args, void *ret_value, int *ret_tcode, void *resource_handle);\n\n"
        )

        # read node attributes
        shapes = None
        for s in graph["attrs"]["shape"]:
            if isinstance(s, str):
                continue
            shapes = s
        dltypes = None
        for t in graph["attrs"]["dltype"]:
            if isinstance(t, str):
                continue
            dltypes = t
        storage_ids = None
        for ids in graph["attrs"]["storage_id"]:
            if isinstance(ids, str):
                continue
            storage_ids = ids

        # read the storage id of the linked parameters
        param_ids = set()

        for i, node in enumerate(graph["nodes"]):
            if node["name"] in params:
                idx = graph["node_row_ptr"][i]
                param_ids.add(storage_ids[idx])

        storage_sizes = {}
        define_tensors = ""
        lookup = "\tTVMValue storage_id; int lookup_arg_type = kTVMArgInt, lookup_ret_type;\n"

        # iterate over tensors
        for i, (shape, dltype) in enumerate(zip(shapes, dltypes)):
            match = re.match(r"(float|u?int)(\d+)", dltype)
            if match:
                if dltype == "float32":
                    ctype = "float"
                    kDL = "kDLFloat"
                elif dltype == "float64":
                    ctype = "double"
                    kDL = "kDLFloat"
                elif match.group(1) != "float":
                    ctype = dltype + "_t"
                    kDL = "kDLUInt" if dltype[0] == "u" else "kDLInt"
                dldatatype = f"{{ {kDL}, {match.group(2)}, 1 }}"
            else:
                print("can not handle this type")
                exit(1)

            storage_id = storage_ids[i]
            if storage_id in param_ids:
                # in the main function, first write pointers for the linked parameters to the tensors
                lookup += f"\tstorage_id.v_int64 = { storage_id };\n"
                lookup += f"\t_lookup_linked_param(&storage_id, &lookup_arg_type, 1, &tensor{i}.data, &lookup_ret_type, NULL);\n"
                data_pointer = "NULL"
            else:
                # calculate the needed size for allocated arrays
                size = (int(match.group(2)) + 7) // 8 * prod(shape)
                old_size = storage_sizes.get(storage_id, 0)
                if size > old_size:
                    storage_sizes[storage_id] = size

                data_pointer = f"data{storage_id}"

            # define the tensors after allocating the arrays
            shape_str = ", ".join(map(str, shape))
            define_tensors += f"int64_t shape{i}[] = {{ {shape_str} }};\n"
            define_tensors += f"DLTensor tensor{i} = {{ { data_pointer }, {{ kDLCPU, 0 }}, {len(shape)}, {dldatatype}, shape{i}, NULL, 0 }};\n\n"

        # allocate memory for remaining tensors
        for storage_id, size in storage_sizes.items():
            file.write(f"char data{storage_id}[{size}];\n\n")

        file.write(define_tensors)

        call_ops = "\tint error = 0;\n"

        # iterate over operator nodes
        for i, node in enumerate(graph["nodes"]):
            if node["op"] == "tvm_op":
                func_name = node["attrs"]["func_name"]
                if func_name == "__nop":
                    continue
                inputs = node["inputs"]
                num_inputs = int(node["attrs"]["num_inputs"])
                num_outputs = int(node["attrs"]["num_outputs"])
                num_args = num_inputs + num_outputs

                # allocate argument array
                args = [graph["node_row_ptr"][n] + j for n, j, _ in inputs]
                args += [graph["node_row_ptr"][i] + j for j in range(num_outputs)]
                args_str = ", ".join(f"{{ .v_handle = &tensor{arg} }}" for arg in args)
                file.write(f"\nTVMValue args{i}[] = {{ {args_str} }};\n")

                # allocate argument types
                types = ", ".join(["kTVMDLTensorHandle"] * len(args))
                file.write(f"int types{i}[] = {{ {types} }};\n")

                # declare this nodes function
                file.write(
                    f"int {func_name}(TVMValue* args, int* type_codes, int num_args);\n"
                )

                # call function and check for errors
                call_ops += f"\terror = {func_name}(args{i}, types{i}, {num_args});\n"
                call_ops += "\tif(error) return error;\n"

        # write the main function
        file.write("\nint run() {\n")
        file.write(lookup)
        file.write(call_ops)
        file.write("\treturn 0;\n")
        file.write("}\n")

    # def build(self, module, target, pulp_key):
    #     if target == "llvm":
    #         if pulp_key:
    #             tgt = "llvm -mtriple=riscv32-unknown-elf -mcpu=generic-rv32 -mattr=+m,+c,+xpulpv -runtime=c -system-lib=1 -keys=pulp"
    #         else:
    #             tgt = "llvm -mtriple=riscv32-unknown-elf -mcpu=generic-rv32 -mattr=+m,+c,+xpulpv -runtime=c -system-lib=1"
    #         with tvm.transform.PassContext(opt_level=3):
    #             m = relay.build(module, tgt)

    #     else:
    #         if pulp_key:
    #             tgt = "c -mcpu=generic-rv32 -runtime=c -system-lib -keys=pulp"
    #         else:
    #             tgt = "c -mcpu=generic-rv32 -runtime=c -system-lib"
    #         with tvm.transform.PassContext(
    #             opt_level=3, config={"tir.disable_vectorize": True}
    #         ):
    #             m = relay.build(module, tgt)

    #     lib = m.module
    #     graph = m.graph_json
    #     with open("build/runner.c", "w") as file:
    #         g = json.loads(graph)
    #         self.run(g, file)
    #     lib.export_library("build/model.tar")
    #     for i, m in enumerate(lib._collect_dso_modules()):
    #         with open("build/lib" + str(i) + ".ll", "w") as file:
    #             file.write(m.get_source())
    #     with open("build/graph.json", "w") as file:
    #         file.write(graph)
