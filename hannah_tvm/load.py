import logging
import sys

from pathlib import Path

import onnx
import torch
import tvm
import tvm.relay as relay
import numpy as np
import tensorflow as tf
import tvm.relay.testing.tf as tf_testing
try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

from hydra.utils import to_absolute_path

logger = logging.getLogger("hannah-tvm.compile")


def _load_torch(model_path, input_shapes):
    logger.info("Loading model %s", str(model_path))

    script_model = torch.script.load(model_path)


def _load_onnx(model_path, input_shapes):
    logger.info("Loading model %s", str(model_path))
    try:
        import onnx
        import onnx.version_converter
    except:
        logger.error("Could not import onnx")
        sys.exit(-1)

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    # inferred_model = onnx.version_converter.convert_version(inferred_model, 11)

    graph = inferred_model.graph

    type_map = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.INT32: "int32",
        onnx.TensorProto.INT16: "int16",
        onnx.TensorProto.INT8: "int8",
        onnx.TensorProto.UINT64: "uint64",
        onnx.TensorProto.UINT32: "uint32",
        onnx.TensorProto.UINT16: "uint16",
        onnx.TensorProto.UINT8: "uint8",
    }

    shape_dict = {}
    dtype_dict = {}

    initializer_names = [init.name for init in graph.initializer]
    for input in graph.input:
        if input.name in initializer_names:
            continue

        tensor_type = input.type.tensor_type
        input_shape = [d.dim_value for d in tensor_type.shape.dim]

        shape_dict[input.name] = tuple(input_shape)
        dtype_dict[input.name] = type_map[input.type.tensor_type.elem_type]

    print("loading onnx")
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, dtype_dict)

    input_data = {}
    for name, shape in shape_dict.items():
        input_data[name] = np.random.uniform(size=shape).astype(dtype_dict[name])

    return mod, params, input_data


def _load_tflite(model_path, input_shapes):

    logger.info("Loading model %s", str(model_path))

    try:
        import tflite
        from tflite.TensorType import TensorType as TType
    except ModuleNotFoundError:
        logger.error("Could not import tflite")
        sys.exit(-1)

    class TensorInfo:
        def __init__(self, t):
            self.name = t.Name().decode()

            typeLookup = {
                TType.FLOAT32: (4, "float32"),
                TType.UINT8: (1, "uint8"),
                TType.INT8: (1, "int8"),
            }
            self.tysz, self.ty = typeLookup[t.Type()]
            assert self.ty != ""

            shape = tuple([t.Shape(si) for si in range(0, t.ShapeLength())])
            self.shape = shape

            self.size = self.tysz
            for dimSz in self.shape:
                self.size *= dimSz

    class ModelInfo:
        def __init__(self, model):
            assert model.SubgraphsLength() == 1
            g = model.Subgraphs(0)

            self.inTensors = []
            for i in range(0, g.InputsLength()):
                t = g.Tensors(g.Inputs(i))
                self.inTensors.append(TensorInfo(t))

            self.outTensors = []
            for i in range(0, g.OutputsLength()):
                t = g.Tensors(g.Outputs(i))
                self.outTensors.append(TensorInfo(t))

    modelBuf = open(model_path, "rb").read()

    tflModel = tflite.Model.GetRootAsModel(modelBuf, 0)

    shapes = {}
    types = {}

    modelInfo = ModelInfo(tflModel)
    for t in modelInfo.inTensors:
        logger.info(f'Input, "{t.name}" {t.ty} {t.shape}')
        shapes[t.name] = t.shape
        types[t.name] = t.ty

    mod, params = relay.frontend.from_tflite(
        tflModel, shape_dict=shapes, dtype_dict=types
    )

    inputs = {}
    for tensor_info in modelInfo.inTensors:
        inputs[t.name] = np.random.uniform(size=t.shape).astype(t.ty)

    return mod, params, inputs


def _load_tensorflow(model_path, input_shapes):

    with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        node_names = [n.name for n in graph_def.node]

        with tf_compat_v1.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(sess, node_names[-1])

        all_placeholders = [placeholder for op in tf_compat_v1.get_default_graph().get_operations() if op.type=='Placeholder' for placeholder in op.values()]

    shapes = {}
    types = {}

    for input in all_placeholders:
        shapes[input.op.name] = tuple(input.shape)
        types[input.op.name] = input.dtype.as_numpy_dtype

    mod, params = relay.frontend.from_tensorflow(graph_def, shape=shapes) # layout="NCHW"

    inputs = {}
    for input in all_placeholders:
        inputs[input.op.name] = np.random.uniform(size=shapes[input.op.name]).astype(types[input.op.name])

    return mod, params, inputs


def load_model(model):
    model_path = Path(to_absolute_path(model.file))
    input_shapes = model.input_shapes

    if model_path.suffix == ".onnx":
        return _load_onnx(model_path, input_shapes)
    elif model_path.suffix == ".pt":
        return _load_torch(model_path, input_shapes)
    elif model_path.suffix == ".tflite":
        return _load_tflite(model_path, input_shapes)
    elif model_path.suffix == ".pb":
        return _load_tensorflow(model_path, input_shapes)
    else:
        raise Exception(f"File format not supported {model_path.suffix}")
