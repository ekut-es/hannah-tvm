#
# Copyright (c) 2023 hannah-tvm contributors.
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

import tvm
from hannah.callbacks.backends import InferenceBackendBase

from .backend import build_relay


class UMAUltraTrailBackend(InferenceBackendBase):
    """Inference backend for UltraTrail"""

    def __init__(
        self,
        backend_dir: str,
        limit_batch_size: int = 1,
        val_batches: int = 0,
        test_batches: int = 1,
        val_frequency: int = 1,
    ) -> None:
        """Instantiate the UltraTrail backend

        Args:
            backend_dir (str): Path to the UltraTrail UMA backend.
            val_batches (int, optional): Number of batches to run through the backend on validation. Defaults to 0.
            test_batches (int, optional): Number of batches to run through the backend on testing. Defaults to 1.
            val_frequency (int, optional): Validate on every n batches. Defaults to 1.
        """
        super().__init__(val_batches, test_batches, val_frequency)
        self.limit_batch_size = limit_batch_size
        self.torch_model = None

        self.backend_dir = backend_dir
        self.ut_backend = None

    def prepare(self, model):
        logging.info("Preparing model for target")
        self.torch_model = copy.deepcopy(model)
        self.torch_model.cpu()
        self.torch_model.eval()

        mod, params = build_relay(
            self.torch_model.model, self.torch_model.example_feature_array
        )
        mod = tvm.relay.transform.InferType()(mod)

        # Load backend package
        sys.path.append(self.backend_dir)
        from uma.backend import UltraTrailBackend  # pytype: disable=import-error

        self.ut_backend = UltraTrailBackend()

        # Set bitwidth with respect to model
        # TODO: Better method for quantization bit determination
        bw_w = self.torch_model.model.linear[0][0].weight_fake_quant.bits
        bw_b = self.torch_model.model.linear[0][0].bias_fake_quant.bits
        bw_f = bw_b
        self.ut_backend.hw_description.set_bw(bw_w, bw_b, bw_f)

        self.ut_backend.register()
        self.ut_backend.partition(mod, params)

    def run_batch(self, inputs=None):
        if inputs is None:
            logging.critical("Backend batch is empty")
            return None

        inputs = inputs.cpu()
        inputs = inputs[: self.limit_batch_size]

        x = self.torch_model._extract_features(inputs)
        x = self.torch_model.normalizer(x)
        y = self.torch_model.model(x)

        xs = x.cpu().split(1)
        ys = y.cpu().split(1)
        ys = [t.squeeze() for t in ys]

        results = self.ut_backend.run(xs, ys)

        return results

    def estimate_metrics(self):
        results = self.ut_backend.predict()
        return results
