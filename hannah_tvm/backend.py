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
import copy
import logging
import sys
from typing import Optional

import numpy as np
import torch
import tvm
import tvm.relay
from hannah.backends.base import InferenceBackendBase
from tvm.auto_scheduler.measure import prepare_input_map
from tvm.contrib import utils
from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER
from tvm.testing.aot import AOTTestModel, compile_and_run, generate_ref_data

from hannah_tvm.connectors import init_board_connector

from . import pass_instrument
from .config import Board, TunerConfig
from .export import build_relay
from .passes.legalize import LegalizeQuantizedTypes
from .task import ModelConfig, TuningTask
from .tracer import QuantizationTracer, RelayConverter

logger = logging.getLogger(__name__)


class TVMBackend(InferenceBackendBase):
    """Inference backend for tvm"""

    def __init__(
        self,
        board: Board,
        tuner: Optional[TunerConfig] = None,
        val_batches: int = 1,
        test_batches: int = 1,
        val_frequency: int = 1,
        tune=False,
    ) -> None:
        """Instantiate the tvm backend for a target board and tuner configuration

        Args:
            board (BoardConfig): Target board description
            tuner (TunerConfig): Tuner configuration
            val_batches (int, optional): Number of batches to run through the backend on validation. Defaults to 1.
            test_batches (int, optional): Number of batches to run through the backend on testing. Defaults to 1.
            val_frequency (int, optional): Validate on every n batches. Defaults to 1.
            tune (bool, optional): Tune the model. Defaults to False.
        """
        super().__init__(val_batches, test_batches, val_frequency, tune)

        self.torch_model = None
        self.board_config = board
        self.tuner_config = tuner
        self.task = None

    def prepare(self, model):
        logging.info("Preparing model for target")
        model = copy.deepcopy(model)
        model.eval()
        model.cpu()

        self.torch_model = model

        mod, params = build_relay(model.model, model.example_feature_array)
        mod = tvm.relay.transform.InferType()(mod)

        try:
            mod = LegalizeQuantizedTypes()(mod)
        except Exception as e:
            logger.warning("Failed to legalize quantized types")
            logger.warning(e)

        self._connector = init_board_connector(self.board_config)

        task_connector = self._connector.task_connector()

        input_names = []
        input_types = []
        for gvar, func in mod.functions.items():
            if gvar.name_hint == "main":
                for param in func.params:
                    if param.name_hint not in params:
                        input_names.append(param.name_hint)
                        input_types.append(param.type_annotation.dtype)

        assert len(input_names) == 1

        input = model.example_feature_array.detach()

        if hasattr(model, "normalizer"):
            input = model.normalizer(input)
        input = input.numpy()
        input = input.astype(input_types[0])

        model_config = ModelConfig(mod, params, {input_names[0]: input})
        model_key = "backend_model"

        self.task = TuningTask(
            board_config=self.board_config,
            model_key=model_key,
            model_config=model_config,
            task_connector=task_connector,
            tuner=self.tuner_config,
        )
        if self.tune:
            self.task.run()

    def export(self):
        self.task.export()

    def characterize(self, model):
        self.prepare(model)

    def run_batch(self, inputs=None):
        if inputs is None:
            logging.critical("Backend batch is empty")
            return None

        device = self.torch_model.device
        self.torch_model.cpu()
        inputs = inputs.cpu()

        features = inputs
        if hasattr(self.torch_model, "features"):
            features = self.torch_model.features(features)

        if hasattr(self.torch_model, "normalizer"):
            features = self.torch_model.normalizer(features)

        mod, params = build_relay(
            self.torch_model.model, self.torch_model.example_feature_array
        )
        mod = tvm.relay.transform.InferType()(mod)

        try:
            mod = LegalizeQuantizedTypes()(mod)
        except Exception as e:
            logger.warning("Failed to legalize quantized types")
            logger.warning(e)

        # FIXME: Make AOT compile and run optional
        # FIXME: Fix hardcoded input and output scaling
        aot_compile = False
        results = []
        for feature in features:
            x = {"x": (feature.numpy() * 2**7).astype("int8")}
            output_list = generate_ref_data(mod, x, params)  # This takes a lot of time
            if aot_compile:
                compile_and_run(
                    AOTTestModel(
                        module=mod, inputs=x, outputs=output_list, params=params
                    ),
                    AOT_DEFAULT_RUNNER,
                    interface_api="c",
                    use_unpacked_api=True,
                )

            results.append(
                torch.tensor(output_list["output"].squeeze().astype("float32"))
                / 2**14
            )

        return torch.stack(results, dim=0)

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
