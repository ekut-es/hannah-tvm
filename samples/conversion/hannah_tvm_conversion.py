#
# Copyright (c) 2023 University of Tübingen.
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
import hannah.conf
import hydra
import torch
import tvm
from hydra.utils import instantiate

from hannah_tvm.passes.legalize import LegalizeQuantizedTypes
from hannah_tvm.tracer import QuantizationTracer, RelayConverter


def convert_to_relay(checkpoint_path, legalize=False, data_folder=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    hparams = checkpoint["hyper_parameters"]

    from pprint import pprint

    pprint(hparams, indent=2)

    if data_folder is not None:
        hparams["dataset"]["data_folder"] = data_folder
    with hydra.initialize_config_module("hannah.conf"):
        module = instantiate(hparams, _recursive_=False)

    module.setup("test")
    module.load_state_dict(checkpoint["state_dict"])

    tracer = QuantizationTracer()

    traced_graph = tracer.trace(module.model)
    converter = RelayConverter(torch.fx.GraphModule(module.model, traced_graph))
    mod, params = converter.run(module.example_feature_array)
    mod = tvm.relay.transform.InferType()(mod)

    if legalize:
        mod = LegalizeQuantizedTypes()(mod)

    return mod, params


if __name__ == "__main__":
    import sys

    checkpoint = sys.argv[1]
    data_folder = sys.argv[2]

    mod, params = convert_to_relay(checkpoint, data_folder=data_folder)
    print(mod)
    # print(params)
