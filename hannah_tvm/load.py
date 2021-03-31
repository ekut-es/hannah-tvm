from pathlib import Path

import onnx
import torch 
import tvm
import tvm.relay as relay
import numpy as np 

from hydra.utils import to_absolute_path

def _load_torch(model_path):
    pass

def _load_onnx(model_path):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    graph = inferred_model.graph

    input = graph.input[0]
    tensor_type = input.type.tensor_type
    input_shape = [d.dim_value for d in tensor_type.shape.dim]

    shape_dict = {input.name: tuple(input_shape)}

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    input_data = {input.name: np.random.uniform(input_shape)}

    return mod, params, input_data

def load_model(model_path):
    model_path = Path(to_absolute_path(model_path))
    
    if model_path.suffix == '.onnx':
        return _load_onnx(model_path)
    elif model_path.suffix == '.torch':
        return _load_torch(model_path)
    else:
        raise Exception(f"File format not supported {model_path.suffix}")