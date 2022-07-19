#
# Copyright (c) 2022 University of Tübingen.
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
import copy
import logging
import sys

import numpy as np
import torch
import tvm
import tvm.relay
from hannah.callbacks.backends import InferenceBackendBase
from tvm.auto_scheduler.measure import prepare_input_map
from tvm.contrib import utils

from hannah_tvm.connectors import init_board_connector

from . import pass_instrument
from .passes.legalize import LegalizeQuantizedTypes
from .task import ModelConfig, TuningTask
from .tracer import QuantizationTracer, RelayConverter


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
        val_batches: int = 1,
        test_batches: int = 1,
        val_frequency: int = 1,
        board=None,
        tuner=None,
    ) -> None:
        """Instantiate the tvm backend for

        Args:
            val_batches (int, optional): Number of batches to run through the backend on validation. Defaults to 1.
            test_batches (int, optional): Number of batches to run through the backend on testing. Defaults to 1.
            val_frequency (int, optional): Validate on every n batches. Defaults to 1.
            board (OmegaconfDict[str, BoardConfig], optional): Dict of target board descriptions. Defaults to None.
        """
        super().__init__(val_batches, test_batches, val_frequency)

        self.torch_model = None
        self.board_config = board
        self.tuner_config = tuner

    def prepare(self, model):
        logging.info("Preparing model for target")
        model = copy.deepcopy(model)
        model.cpu()

        self.torch_model = model

        mod, params = build_relay(model.model, model.example_feature_array)

        mod = tvm.relay.transform.InferType()(mod)
        mod = LegalizeQuantizedTypes()(mod)

        assert len(self.board_config) == 1

        self._connector = init_board_connector(list(self.board_config.values())[0])

        task_connector = self._connector.task_connector()

        model_config = ModelConfig(mod, params, model.example_feature_array)
        model_key = "backend_model"

        for board_key, board_config in self.board_config.items():
            task = TuningTask(
                board_config=board_config,
                board_key=board_key,
                model_key=model_key,
                model_config=model_config,
                task_connector=task_connector,
                tuner=self.tuner_config,
            )
            task.run()

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

        out = None
        results = None

        return out, results

    def __getstate__(self):
        "Do not pickle and copy auto generated values"
        state = self.__dict__

        if "_connector" in state:
            state.pop("_connector")
        if "_task" in state:
            state.pop("_task")
        if "torch_model" in state:
            state.pop("torch_model")

        return state
