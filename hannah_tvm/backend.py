import logging

import torch
import tvm
import tvm.relay

from hannah.callbacks.backends import InferenceBackendBase
from .tracer import QuantizationTracer, RelayConverter
from .passes.legalize import LegalizeQuantizedTypes


class TVMBackend(InferenceBackendBase):
    """Inference backend for tvm"""

    def __init__(self, val_batches=1, test_batches=1, val_frequency=1):
        super().__init__(val_batches, test_batches, val_frequency)

        self.torch_model = None
        self.model = None
        self.params = None
        self.lib = None

    def prepare(self, model):
        logging.info("Preparing model for target")
        self.torch_model = model

        device = model.device

        tracer = QuantizationTracer()

        model.cpu()

        traced_graph = tracer.trace(model.model)
        converter = RelayConverter(torch.fx.GraphModule(model.model, traced_graph))
        mod, params = converter.run(model.example_feature_array)
        mod = tvm.relay.transform.InferType()(mod)
        mod = LegalizeQuantizedTypes()(mod)

        target = "llvm"
        with tvm.transform.PassContext(
            opt_level=3, config={"tir.disable_vectorize": True}
        ):
            lib = tvm.relay.build(mod, target=target, params=params)

        self.model = mod
        self.params = params
        self.lib = lib

        model.to(device)

    def run_batch(self, inputs=None):
        if inputs is None:
            logging.critical("Backend batch is empty")
            return None

        device = self.torch_model.device
        self.torch_model.cpu()
        inputs = inputs.cpu()

        feature = self.torch_model.features(inputs)
        feature = self.torch_model.normalizer(feature)

        features = torch.split(feature, 1)

        features = [x.detach().cpu().numpy() for x in features]
        results = []

        for input in features:
            from tvm.contrib import utils
            import numpy as np

            input = input * 128
            input = input.round()
            input = np.clip(input, -128, 127)

            temp = utils.tempdir()
            path_lib = temp.relpath("deploy_lib.tar")
            self.lib.export_library(path_lib)
            print(temp.listdir())

            loaded_lib = tvm.runtime.load_module(path_lib)
            input_data = tvm.nd.array(input)

            module = tvm.contrib.graph_executor.GraphModule(
                loaded_lib["default"](tvm.cpu())
            )
            module.run(data=input_data)
            out_deploy = module.get_output(0).numpy()

            # Print first 10 elements of output
            print(out_deploy.flatten()[0:10])

            out = out_deploy.astype(float)
            out = out / (2 ** 14)

            print(out[0:10])

            results.append(torch.from_numpy(out))

        out = torch.stack(results).squeeze(1)
        self.torch_model.to(device)

        return out
