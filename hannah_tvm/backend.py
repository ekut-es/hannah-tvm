import logging
from hannah_tvm.experiment_scheduler import BackendExperimentScheduler

import torch
import tvm
from tvm.auto_scheduler.measure import prepare_input_map
import tvm.relay
import numpy as np

from hannah.callbacks.backends import InferenceBackendBase
from .tracer import QuantizationTracer, RelayConverter
from .passes.legalize import LegalizeQuantizedTypes
from . import pass_instrument


class TVMBackend(InferenceBackendBase):
    """Inference backend for tvm"""

    def __init__(self, val_batches=1, test_batches=1, val_frequency=1, board=None, print_after=[], time_passes=False):
        super().__init__(val_batches, test_batches, val_frequency)

        self.torch_model = None
        self.model = None
        self.params = None
        self.lib = None
        self.print_after = print_after
        self.time_passes = time_passes
        self.board_config = board

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

        scheduler = BackendExperimentScheduler({'board': self.board_config, 'n_jobs': 0}, mod, params, {'x': model.example_feature_array.detach().numpy().astype(np.int8)})
        results = scheduler.run()

        return results

        # target = "llvm"
        # instruments = []
        # timing_instrument = None

        # if self.time_passes:
        #     timing_instrument = tvm.ir.instrument.PassTimingInstrument()
        #     instruments.append(timing_instrument)

        # if self.print_after:
        #     instruments.append(pass_instrument.PrintIR(self.print_after))

        # with tvm.transform.PassContext(
        #     opt_level=3, config={"tir.disable_vectorize": True}, instruments=instruments 
        # ):
        #     mod = LegalizeQuantizedTypes()(mod)
        #     lib = tvm.relay.build(mod, target=target, params=params)
            
        #     if timing_instrument:
        #         logging.info("Pass profiles:\n%s", timing_instrument.render())

        # self.model = mod
        # self.params = params
        # self.lib = lib

        # model.to(device)

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
