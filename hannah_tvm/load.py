import logging
import sys

from pathlib import Path

import onnx
import torch
import tvm
import tvm.relay as relay
import numpy as np

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

    input = graph.input[0]
    tensor_type = input.type.tensor_type
    input_shape = [d.dim_value for d in tensor_type.shape.dim]

    shape_dict = {input.name: tuple(input_shape)}

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    input_data = {input.name: np.random.uniform(size=input_shape)}

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
    for name, shape in shapes.items():
        inputs[name] = np.random.uniform(size=shape)

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
    else:
        raise Exception(f"File format not supported {model_path.suffix}")
