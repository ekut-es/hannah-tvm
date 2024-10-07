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
from hannah.backends.base import AbstractBackend
from hannah.modules.base import ClassifierModule
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


class TVMBackend(AbstractBackend):
    """Inference backend for tvm"""

    def __init__(
        self,
        board: Board,
        tuner: Optional[TunerConfig] = None,
    ) -> None:
        """Instantiate the tvm backend for a target board and tuner configuration

        Args:
            board (BoardConfig): Target board description
            tuner (TunerConfig): Tuner configuration
        """
        super().__init__()

        self.torch_model = None
        self.board_config = board
        self.tuner_config = tuner
        self.task = None
        
    def available(self) -> bool:
        return True

    def prepare(self, module: ClassifierModule):
        logging.info("Preparing model for target")
        
        model = copy.deepcopy(module)
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
        
        self.task.run()


    def run(self, *inputs):
        for input in inputs:
            # Create ndarray from tensor
            input_nd = tvm.nd.array(input.numpy(), device=self.task.device())
            
               
            
    def profile(self, *inputs):
        return self.run(*inputs)

    
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
