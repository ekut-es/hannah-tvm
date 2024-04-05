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
import logging

import torch
import tvm

from .tracer import QuantizationTracer, RelayConverter


def remove_dropout(gm):
    for node in gm.graph.nodes:
        if node.op == "call_function":
            if node.target == torch.nn.functional.dropout:
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)

    gm.recompile()
    gm.graph.lint()

    return gm


def build_relay(model, dummy_input):
    try:
        if not isinstance(model, torch.fx.graph_module.GraphModule):
            tracer = QuantizationTracer()
            traced_graph = tracer.trace(model)
            graph_module = torch.fx.GraphModule(model, traced_graph)
        else:
            graph_module = model
        converter = RelayConverter(graph_module)
        mod, params = converter.run(dummy_input)
    except Exception as e:
        logging.warning(
            "Failed to convert model to relay, using fx converter trying with legacy converter"
        )

        if isinstance(model, torch.fx.graph_module.GraphModule):
            model = remove_dropout(model)

        model.cpu()
        dummy_input.cpu()

        script_module = torch.jit.trace(model, dummy_input)

        mod, params = tvm.relay.frontend.from_pytorch(
            script_module,
            [("input", (dummy_input.shape, "float"))],
            use_parser_friendly_name=True,
            keep_quantized_weight=True,
        )

    return mod, params


def export_relay(model, dummy_input, model_file="model.relay", param_file="params.bin"):
    mod, params = build_relay(model, dummy_input)

    mod_txt = mod.astext()
    with open("model.relay", "w") as f:
        f.write(mod_txt)

    params_bin = tvm.runtime.save_param_dict(params)
    with open("params.bin", "wb") as f:
        f.write(params_bin)
