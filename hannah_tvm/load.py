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
import logging
import os
import sys

import appdirs
import fsspec
import numpy as np
import tvm
import tvm.relay as relay
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

logger = logging.getLogger("hannah-tvm.compile")

cache_dir = appdirs.user_cache_dir("hannah-tvm")


def _load_torch(model_path, input_shapes):
    logger.info("Loading model %s", str(model_path))

    try:
        import torch
    except ImportError:
        logger.error("Could not import torch, please make sure it is installed")
        sys.exit(-1)

    script_model_file = fsspec.open(model_path, "rb")
    with script_model_file as f:
        script_model = torch.jit.load(f)

    input_info = []
    for name, shape in input_shapes:
        input_info.append((name, tuple(shape)))

    mod, params = relay.frontend.from_pytorch(script_model, input_info)

    input_data = {}
    for name, shape in input_info:
        input_data[name] = np.random.uniform(size=shape).astype(np.float32)

    return mod, params, input_data


def _load_onnx(model_path, input_shapes):
    logger.info("Loading model %s", str(model_path))
    try:
        import onnx
        import onnx.version_converter
    except ImportError:
        logger.error("Could not import onnx, please make sure it is installed")
        sys.exit(-1)

    model_file = fsspec.open(
        "filecache::" + model_path,
        filecache={"cache_storage": os.path.join(cache_dir, "models")},
    )
    with model_file as f:
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
        # onnx_model = onnx.version_converter.convert_version(onnx_model, 11)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    graph = onnx_model.graph

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
        raise Exception("Could not import tflite")

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

            self.in_tensors = []
            for i in range(0, g.InputsLength()):
                t = g.Tensors(g.Inputs(i))
                self.in_tensors.append(TensorInfo(t))

            self.out_tensors = []
            for i in range(0, g.OutputsLength()):
                t = g.Tensors(g.Outputs(i))
                self.out_tensors.append(TensorInfo(t))

    model_file = fsspec.open(model_path, "rb")

    with model_file as f:
        model_buf = f.read()
        tflite_model = tflite.Model.GetRootAsModel(model_buf, 0)

    shapes = {}
    types = {}

    model_info = ModelInfo(tflite_model)
    for t in model_info.in_tensors:
        logger.info(f'Input, "{t.name}" {t.ty} {t.shape}')
        shapes[t.name] = t.shape
        types[t.name] = t.ty

    mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict=shapes, dtype_dict=types
    )

    inputs = {}
    for tensor_info in model_info.in_tensors:
        inputs[t.name] = np.random.uniform(size=t.shape).astype(t.ty)

    return mod, params, inputs


def _load_tensorflow(model_path, input_shapes):
    import tvm.relay.testing.tf as tf_testing

    try:
        import tensorflow as tf
    except ImportError:
        logging.error("Could not import tensorflow, please make sure it is installed")
        sys.exit(-1)

    try:
        tf_compat_v1 = tf.compat.v1
    except ImportError:
        tf_compat_v1 = tf

    with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        node_names = [n.name for n in graph_def.node]

        with tf_compat_v1.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(sess, node_names[-1])

        all_placeholders = [
            placeholder
            for op in tf_compat_v1.get_default_graph().get_operations()
            if op.type == "Placeholder"
            for placeholder in op.values()
        ]

    shapes = {}
    types = {}

    for input in all_placeholders:
        shapes[input.op.name] = tuple(input.shape)
        types[input.op.name] = input.dtype.as_numpy_dtype

    mod, params = relay.frontend.from_tensorflow(graph_def, shape=shapes)

    inputs = {}
    for input in all_placeholders:
        inputs[input.op.name] = np.random.uniform(size=shapes[input.op.name]).astype(
            types[input.op.name]
        )

    return mod, params, inputs


def load_model(model):
    model_path = model.url
    input_shapes = model.input_shapes
    filename = model.filename

    if filename is not None:
        suffix = filename.split(".")[-1]
    else:
        suffix = model_path.split(".")[-1]

    if suffix == "onnx":
        return _load_onnx(model_path, input_shapes)
    elif suffix == "pt":
        return _load_torch(model_path, input_shapes)
    elif suffix == "tflite":
        return _load_tflite(model_path, input_shapes)
    elif suffix == "pb":
        return _load_tensorflow(model_path, input_shapes)
    else:
        raise Exception(f"File format not supported {suffix}")
