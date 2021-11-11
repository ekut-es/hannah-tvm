import logging
import copy
from hannah_tvm.experiment_scheduler import BackendScheduler

import torch
import tvm
from tvm.auto_scheduler.measure import prepare_input_map
import tvm.relay
import numpy as np

from hannah.callbacks.backends import InferenceBackendBase
from .tracer import QuantizationTracer, RelayConverter
from .passes.legalize import LegalizeQuantizedTypes
from . import pass_instrument


def build_relay(model, dummy_input):
    tracer = QuantizationTracer()

    traced_graph = tracer.trace(model)
    converter = RelayConverter(torch.fx.GraphModule(model, traced_graph))
    mod, params = converter.run(dummy_input)

    return mod, params


def export_relay(model, dummy_input, model_file="model.relay", param_file="params.bin"):
    mod, params = build_relay(model, dummy_input)

    mod_txt = mod.astext()
    with open("model.relay", "w") as f:
        f.write(mod_txt)

    params_bin = tvm.runtime.save_param_dict(params)
    with open("params.bin", "wb") as f:
        f.write(params_bin)


class TVMBackend(InferenceBackendBase):
    """Inference backend for tvm"""

    def __init__(
        self,
        val_batches=1,
        test_batches=1,
        val_frequency=1,
        board=None,
        print_after=[],
        time_passes=False,
        tune=False,
    ):
        """Instantiate the tvm backend for

        Args:
            val_batches (int, optional): Number of batches to run through the backend on validation. Defaults to 1.
            test_batches (int, optional): Number of batches to run through the backend on testing. Defaults to 1.
            val_frequency (int, optional): Validate on every n batches. Defaults to 1.
            board (OmegaconfDict[str, BoardConfig], optional): Dict of target board descriptions. Defaults to None.
            print_after (list, optional): list of tvm pass names, prints IR after corresponding passes. Defaults to [].
            time_passes (bool, optional): Extract list of timed values for each pass. Defaults to False.
            tune (bool, optional): Run autotuning before execution on target. Defaults to False.
        """
        super().__init__(val_batches, test_batches, val_frequency)

        self.torch_model = None
        self.model = None
        self.params = None
        self.lib = None
        self.print_after = print_after
        self.time_passes = time_passes
        self.board_config = board
        self.tune = tune

    def prepare(self, model):
        logging.info("Preparing model for target")
        model = copy.deepcopy(model)
        model.cpu()

        self.torch_model = model

        device = model.device

        mod, params = build_relay(model.model, model.example_feature_array)

        mod = tvm.relay.transform.InferType()(mod)
        mod = LegalizeQuantizedTypes()(mod)

        scheduler = BackendScheduler(
            {"board": self.board_config, "n_jobs": 0},
            mod,
            params,
            {"x": model.example_feature_array.detach().numpy().astype(np.int8)},
        )

    def characterize(self, model):
        self.prepare(model)

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

        return out, results

    def __getstate__(self):
        "Do not pickle and copy auto generated values"
        state = self.__dict__

        state.pop("torch_model")
        state.pop("params")
        state.pop("lib")

        return state
