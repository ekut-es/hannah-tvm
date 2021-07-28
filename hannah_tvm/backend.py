import logging

import torch
import tvm
from tvm import relay
from tvm.contrib import graph_executor

try:
    from speech_recognition.callbacks.backends import InferenceBackendBase
    from speech_recognition.models.factory.qat import QAT_MODULE_MAPPINGS
except:
    pass

from .quantize import quantize
from .optimize import pre_quantize_opts


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
