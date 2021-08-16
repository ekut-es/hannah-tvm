import logging

import torch
import tvm
from tvm import relay
from tvm.contrib import graph_executor

from hannah.callbacks.backends import InferenceBackendBase
from .tracer import QuantizationTracer, RelayConverter


class TVMBackend(InferenceBackendBase):
    """Inference backend for tvm"""

    def __init__(self, val_batches=1, test_batches=1, val_frequency=1):
        super().__init__(val_batches, test_batches, val_frequency)

        if tvm is None:
            raise Exception(
                "No tvm installation found please make sure that hannah-tvm is installed"
            )

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


class TVMBackend(InferenceBackendBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tvm_model = None
        self.tvm_params = None
        self.tvm_ctx = None
        self.tvm_lib = None

        self.module = None

    def prepare(self, module):
        # module = copy.deepcopy(module)
        model = module.model

        if hasattr(model, "qconfig"):
            model = torch.quantization.convert(
                model, mapping=QAT_MODULE_MAPPINGS, remove_qconfig=True
            )
            module.model = model

        print(model)

        self.module = module

        scripted_model = torch.jit.trace(
            model, module.example_feature_array.to(module.device)
        ).eval()
        shape_list = [("input0", module.example_feature_array.shape)]

        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        # mod = pre_quantize_opts(mod, params)
        # mod = quantize(mod, params)

        self.tvm_model = mod
        self.tvm_params = params

        target = "llvm"
        target_host = "llvm"
        ctx = tvm.cpu(0)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(
                mod, target=target, target_host=target_host, params=params
            )

        self.tvm_ctx = ctx
        self.tvm_lib = lib

    def run_batch(self, inputs=None):
        assert self.module is not None
        if inputs is None:
            logging.critical("Backend batch is empty")
            return None

        m = graph_executor.GraphModule(self.tvm_lib["default"](self.tvm_ctx))
        inputs = self.module._extract_features(inputs.cuda())
        inputs = self.module.normalizer(inputs)

        splitted_inputs = torch.split(inputs, 1)

        splitted_results = []
        for input in splitted_inputs:
            m.set_input("input0", tvm.nd.array(input.cpu().numpy(), self.tvm_ctx))
            m.run()
            res = m.get_output(0)
            splitted_results.append(torch.squeeze(torch.tensor(res.asnumpy()), 0))

        result = torch.stack(splitted_results)
        result = result.to(inputs.device)

        return result
