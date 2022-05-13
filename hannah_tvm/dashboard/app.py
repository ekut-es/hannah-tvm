import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, Input, Output, dcc, html

from ..dataset import DatasetFull


def main():
    app = Dash("Hanna-TVM Results")

    dataset = DatasetFull()
    measurements = dataset.measurements()
    models = measurements["Model"].unique()
    boards = measurements["Board"].unique()

    app.layout = html.Div(
        children=[
            html.H1(children="Tuning Results"),
            html.H2(children="Overview"),
            dcc.Dropdown(["cuda", "llvm"], "llvm", id="target-selection"),
            dcc.Dropdown(
                ["Duration StdDev", "Duration PtP"],
                "Duration StdDev",
                id="error-selection",
            ),
            dcc.Dropdown(
                models,
                [x for x in models if x != "sine"],
                multi=True,
                id="model-selection",
            ),
            dcc.Dropdown(boards, boards, multi=True, id="board-selection"),
            dcc.RadioItems(
                ["Absolute", "Speedup"], "Absolute", inline=True, id="type-selection"
            ),
            dcc.Graph(id="overview-graph"),
        ]
    )

    @app.callback(
        Output("overview-graph", "figure"),
        Input("target-selection", "value"),
        Input("error-selection", "value"),
        Input("model-selection", "value"),
        Input("board-selection", "value"),
        Input("type-selection", "value"),
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

    app.run_server(debug=True)
