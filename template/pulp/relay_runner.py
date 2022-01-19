
import tvm
from tvm import relay, topi
import json
import sys


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

    device = "{ kDLCPU, 0 }"
    for i, (shape, dltype) in enumerate(zip(shapes, dltypes)):
        if dltype == "float32":
            ctype = "float"
            dldatatype = "{ kDLFloat, 32, 1 }"
        elif dltype == "int32":
            ctype = "int32_t"
            dldatatype = "{ kDLInt, 32, 1 }"
        elif dltype == "int16":
            ctype = "int16_t"
            dldatatype = "{ kDLInt, 16, 1 }"
        elif dltype == "int8":
            ctype = "int8_t"
            dldatatype = "{ kDLInt, 8, 1 }"
        elif dltype == "uint32":
            ctype = "uint32_t"
            dldatatype = "{ kDLUInt, 32, 1 }"
        elif dltype == "uint16":
            ctype = "uint16_t"
            dldatatype = "{ kDLUInt, 16, 1 }"
        elif dltype == "uint8":
            ctype = "uint8_t"
            dldatatype = "{ kDLUInt, 8, 1 }"
        else:
            print("can not handle this type")
            exit(1)

        file.write("#define dtype%i %s" % (i, ctype))
        file.write("\n%s data%i[%s];\n" % (ctype, i, "][".join(str(x) for x in shape)))
        file.write("int64_t shape%i[] = { %s };\n"
                   % (i, ", ".join(str(x) for x in shape)))
        file.write("DLTensor tensor%i = { data%i, %s, %i, %s, shape%i, NULL, 0 };\n"
                   % (i, i, device, len(shape), dldatatype, i))

    for i, node in enumerate(graph["nodes"]):
        if node["op"] == "tvm_op":
            inputs = node["inputs"]
            num_outputs = int(node["attrs"]["num_outputs"])
            args = [graph["node_row_ptr"][n] + j for n, j, _ in inputs]
            args += [graph["node_row_ptr"][i] + j for j in range(num_outputs)]
            args = ", ".join("{ .v_handle = &tensor%i }" % arg for arg in args)
            file.write("\nTVMValue args%i[] = { %s };\n" % (i, args))
            file.write("int types%i[] = { %s };\n"
                       % (i, (", kTVMDLTensorHandle" * (len(inputs) + num_outputs))[2:] ))
            file.write("int %s(TVMValue* args, int* type_codes, int num_args);\n"
                       % node["attrs"]["func_name"])

    file.write("\nint run() {\n")
    file.write("\tint error = 0;\n")
    for i, node in enumerate(graph["nodes"]):
        if node["op"] == "tvm_op":
            attrs = node["attrs"]
            file.write("\terror = %s(args%i, types%i, %i);\n"
                       % (attrs["func_name"], i, i,
                          int(attrs["num_inputs"]) + int(attrs["num_outputs"])))
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

