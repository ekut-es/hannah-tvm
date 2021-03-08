import logging
import torch

import tvm
from tvm import relay
from tvm.contrib import graph_runtime

from speech_recognition.callbacks.backends import InferenceBackendBase


class TVMBackend(InferenceBackendBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tvm_model = None
        self.tvm_params = None
        self.tvm_ctx = None
        self.tvm_lib = None

        self.module = None

    def prepare(self, module):
        self.module = module
        scripted_model = torch.jit.trace(module.model, module.example_feature_array).eval()
        shape_list = [('input0', module.example_feature_array.shape)]
        
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        self.tvm_model = mod
        self.tvm_params = params 

        target = "llvm"
        target_host = "llvm"
        ctx = tvm.cpu(0)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, target_host=target_host, params=params)

        self.tvm_ctx = ctx
        self.tvm_lib = lib 


    def run_batch(self, inputs=None):
        assert self.module is not None 
        if inputs is None:
            logging.critical("Backend batch is empty")
            return None


        m = graph_runtime.GraphModule(self.tvm_lib["default"](self.tvm_ctx))
        inputs = self.module._extract_features(inputs)
        inputs = self.module.normalizer(inputs)

        splitted_inputs = torch.split(inputs, 1)

        splitted_results = []
        for input in splitted_inputs:
            m.set_input('input0', tvm.nd.array(input.cpu().numpy(), self.tvm_ctx))
            m.run()
            res = m.get_output(0)
            splitted_results.append(torch.squeeze(torch.tensor(res.asnumpy()), 0))

        result = torch.stack(splitted_results)
        result = result.to(inputs.device)
        
        return result