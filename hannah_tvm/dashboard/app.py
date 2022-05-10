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
            dcc.Graph(id="overview-graph"),
        ]
    )

    @app.callback(
        Output("overview-graph", "figure"),
        Input("target-selection", "value"),
        Input("error-selection", "value"),
        Input("model-selection", "value"),
    )
    def update_overview_figure(target, error, models):
        print(models)
        overview_fig = go.Figure()
        for scheduler in ["baseline", "autotvm", "auto_scheduler"]:
            filtered_measurements = measurements.query(
                "Tuner == @scheduler and Target == @target and Model == @models"
            )
            overview_fig.add_trace(
                go.Bar(
                    x=[filtered_measurements["Board"], filtered_measurements["Model"]],
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
