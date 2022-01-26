
import tvm
from tvm import relay
import json
import re


def runner(graph, file):

    file.write("#include \"../runner.h\"\n#include \"tvm/runtime/c_runtime_api.h\"\n\n")

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

    for i, (shape, dltype) in enumerate(zip(shapes, dltypes)):
        match = re.match(r"(float|u?int)(\d+)", dltype)
        if match:
            if match.group(1) == "float":
                ctype = "float"
                kDL = "kDLFloat"
            else:
                ctype = dltype + "_t"
                kDL = "kDLUInt" if dltype[0] == "u" else "kDLInt"
            dldatatype = f"{{ {kDL}, {match.group(2)}, 1 }}"
        else:
            print("can not handle this type")
            exit(1)

        file.write(f"#define dtype{i} {ctype}\n")
        file.write(f"{ctype} data{i}{''.join(f'[{x}]' for x in shape)};\n")
        file.write(f"int64_t shape{i}[] = {{ {', '.join(map(str, shape))} }};\n")
        data_pointer = f"&data{i}" if len(shape) == 0 else f"data{i}"
        file.write(f"DLTensor tensor{i} = {{ { data_pointer }, {{ kDLCPU, 0 }}, {len(shape)}, {dldatatype}, shape{i}, NULL, 0 }};\n")

    for i, node in enumerate(graph["nodes"]):
        if node["op"] == "tvm_op":
            inputs = node["inputs"]
            num_outputs = int(node["attrs"]["num_outputs"])
            args = [graph["node_row_ptr"][n] + j for n, j, _ in inputs]
            args += [graph["node_row_ptr"][i] + j for j in range(num_outputs)]
            args_str = ", ".join(f"{{ .v_handle = &tensor{arg} }}" for arg in args)
            file.write(f"\nTVMValue args{i}[] = {{ {args_str} }};\n")
            file.write(f"int types{i}[] = {{ {', '.join(['kTVMDLTensorHandle'] * len(args))} }};\n")
            file.write(f"int {node['attrs']['func_name']}(TVMValue* args, int* type_codes, int num_args);\n")

    file.write("\nint run() {\n")
    file.write("\tint error = 0;\n")
    for i, node in enumerate(graph["nodes"]):
        if node["op"] == "tvm_op":
            attrs = node["attrs"]
            file.write(f"\terror = {attrs['func_name']}(args{i}, types{i}, {int(attrs['num_inputs']) + int(attrs['num_outputs'])});\n")
            file.write("\tif(error) return error;\n")
    file.write("\treturn 0;\n")
    file.write("}\n")


def build(module, target, pulp_key):
    if target == "llvm":
        if pulp_key:
            tgt = "llvm -mtriple=riscv32-unknown-elf -mcpu=generic-rv32 -mattr=+m,+c,+xpulpv -runtime=c -system-lib=1 -keys=pulp"
        else:
            tgt = "llvm -mtriple=riscv32-unknown-elf -mcpu=generic-rv32 -mattr=+m,+c,+xpulpv -runtime=c -system-lib=1"
        with tvm.transform.PassContext(opt_level=3):
            m = relay.build(module, tgt)

    else:
        if pulp_key:
            tgt = "c -mcpu=generic-rv32 -runtime=c -system-lib -keys=pulp"
        else:
            tgt = "c -mcpu=generic-rv32 -runtime=c -system-lib"
        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize" : True}):
            m = relay.build(module, tgt)

    lib = m.module
    graph = m.graph_json
    with open("build/runner.c", "w") as file:
        g = json.loads(graph)
        runner(g, file)
    lib.export_library("build/model.tar")
    for i, m in enumerate(lib._collect_dso_modules()):
        with open("build/lib" + str(i) + ".ll", "w") as file:
            file.write(m.get_source())
    with open("build/graph.json", "w") as file:
        file.write(graph)

