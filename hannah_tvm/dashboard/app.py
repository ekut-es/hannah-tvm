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
import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, Input, Output, dash_table, dcc, html
from matplotlib.pyplot import xlabel

from hannah_tvm.passes.op_order import calculate_op_order

from ..dataset import DatasetFull

logger = logging.getLogger(__name__)


def main():
    app = Dash("Hannah-TVM Results")

    logger.info("Loading dataset")
    dataset = DatasetFull()
    measurements = dataset.measurements()
    models = list(measurements["Model"].unique())
    if "" in models:
        models.remove("")
    boards = measurements["Board"].unique()
    network_results = dataset.network_results()

    app.layout = html.Div(
        children=[
            html.H1(children="Tuning Results"),
            # Overview
            html.H2(children="Overview"),
            dcc.Dropdown(
                ["Duration StdDev", "Duration PtP"],
                "Duration StdDev",
                id="overview-error-selection",
            ),
            dcc.Dropdown(
                ["c", "cuda", "llvm"],
                "llvm",
                multi=True,
                id="overview-target-selection",
            ),
            dcc.Dropdown(
                models,
                [x for x in models if x != "sine"],
                multi=True,
                id="overview-model-selection",
            ),
            dcc.Dropdown(boards, boards, multi=True, id="overview-board-selection"),
            dcc.RadioItems(
                ["Absolute", "Speedup"],
                "Absolute",
                inline=True,
                id="overview-type-selection",
            ),
            dcc.Graph(id="overview-graph"),
            # Roofline
            html.H2(children="Roofline analysis"),
            # Network Info
            html.H2(children="Network Info"),
            dcc.Dropdown(
                models, models[0] if len(models) > 1 else "", id="network-info-model"
            ),
            dcc.Dropdown(
                boards, boards[0] if len(boards) > 1 else "", id="network-info-board"
            ),
            dcc.Dropdown(["cuda", "llvm"], "llvm", id="network-info-target"),
            dcc.Dropdown(
                ["baseline", "autotvm", "auto_scheduler"],
                "baseline",
                id="network-info-tuner",
            ),
            dcc.Graph(id="network-info-graph"),
            dash_table.DataTable(
                id="network-info-table",
                columns=[
                    {"name": "Layer", "id": "layer"},
                    {"name": "Hash", "id": "hash"},
                    {"name": "Name", "id": "name"},
                    {"name": "Duration (us)", "id": "duration"},
                ],
                data=[],
            ),
        ]
    )

    @app.callback(
        Output("network-info-graph", "figure"),
        Output("network-info-table", "data"),
        Input("network-info-target", "value"),
        Input("network-info-model", "value"),
        Input("network-info-board", "value"),
        Input("network-info-tuner", "value"),
    )
    def update_network_details(target, model, board, tuner):

        network_info_figure = go.Figure()

        selected_result = None
        for result in network_results:
            if (
                result.target == target
                and result.model == model
                and result.tuner == tuner
                and result.board == board
            ):
                if selected_result is not None:
                    logger.critical("Multiple results found")
                selected_result = result

        if selected_result is None:
            return network_info_figure, []

        relay_model = selected_result.relay
        measurement = selected_result.measurement
        call_profile = measurement["calls"]

        op_table = []
        for layer, call in enumerate(call_profile):
            hash = call["Hash"]["string"]
            name = call["Name"]["string"]
            duration = call["Duration (us)"]["microseconds"]
            op_table.append(dict(layer=layer, hash=hash, name=name, duration=duration))

        op_table_frame = pd.DataFrame.from_records(op_table)

        network_info_figure = px.bar(op_table_frame, y="duration", x="layer")

        return network_info_figure, op_table

    @app.callback(
        Output("overview-graph", "figure"),
        Input("overview-target-selection", "value"),
        Input("overview-error-selection", "value"),
        Input("overview-model-selection", "value"),
        Input("overview-board-selection", "value"),
        Input("overview-type-selection", "value"),
    )
    def update_overview_figure(target, error, models, boards, type):
        overview_fig = go.Figure()
        schedulers = ["baseline", "autotvm", "auto_scheduler"]
        if type == "Speedup":
            schedulers = ["autotvm", "auto_scheduler"]
        for scheduler in schedulers:
            filtered_measurements = measurements.query(
                "Tuner == @scheduler and Target == @target and Model == @models and Board == @boards"
            )

            if type == "Speedup":
                baseline_measurements = measurements.query(
                    "Tuner == 'baseline' and Target == @target and Model == @models and Board == @boards"
                )

                merged_measurements = pd.merge(
                    filtered_measurements,
                    baseline_measurements,
                    on=["Target", "Model", "Board"],
                )

                merged_measurements["Speedup"] = (
                    merged_measurements["Duration (us)_y"]
                    / merged_measurements["Duration (us)_x"]
                )
                filtered_measurements = merged_measurements

                overview_fig.add_trace(
                    go.Bar(
                        x=[
                            filtered_measurements["Board"],
                            filtered_measurements["Model"],
                        ],
                        y=filtered_measurements["Speedup"],
                        name=scheduler,
                    )
                )
            else:
                overview_fig.add_trace(
                    go.Bar(
                        x=[
                            filtered_measurements["Board"],
                            filtered_measurements["Model"],
                        ],
                        y=filtered_measurements["Duration (us)"],
                        name=scheduler,
                        error_y={
                            "type": "data",
                            "array": filtered_measurements[error],
                            "visible": True,
                        },
                    )
                )

        return overview_fig

    logger.info("Starting server")
    app.run_server(debug=True)
