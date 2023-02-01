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
from tvm.contrib import relay_viz
from tvm.relay.testing import resnet


class RelayVisualizer:
    def __init__(self):
        # graphviz attributes
        graph_attr = {"color": "red"}
        node_attr = {"color": "blue"}
        edge_attr = {"color": "black"}

        # VizNode is passed to the callback.
        # We want to color NCHW conv2d nodes. Also give Var a different shape.
        def get_node_attr(node):
            if "nn.conv1d" in node.type_name:
                return {"fillcolor": "blue", "style": "filled", "shape": "box"}
            elif "nn.conv2d" in node.type_name:
                return {"fillcolor": "blue", "style": "filled", "shape": "box"}
            elif "nn.conv3d" in node.type_name:
                return {"fillcolor": "blue", "style": "filled", "shape": "box"}
            elif "nn.linear" in node.type_name:
                return {"fillcolor": "blue", "style": "filled", "shape": "box"}
            elif "nn.dense" in node.type_name:
                return {"fillcolor": "blue", "style": "filled", "shape": "box"}
            elif "Var" in node.type_name:
                return {"shape": "ellipse"}
            return {"shape": "box"}

        # Create plotter and pass it to viz. Then render the graph.
        self.dot_plotter = relay_viz.DotPlotter(
            graph_attr=graph_attr,
            node_attr=node_attr,
            edge_attr=edge_attr,
            get_node_attr=get_node_attr,
        )

    def render(self, mod, param, output_path):
        viz = relay_viz.RelayVisualizer(
            mod,
            relay_param=param,
            plotter=self.dot_plotter,
            parser=relay_viz.DotVizParser(),
        )
        viz.render(output_path)
