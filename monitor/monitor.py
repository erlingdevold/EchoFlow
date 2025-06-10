import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from pathlib import Path
import os
import datetime

dir_input = os.getenv("INPUT_DIR", "/app/input")
dir_output = os.getenv("OUTPUT_DIR", "/app/output")
dir_pp_output = os.getenv("PP_OUTPUT_DIR", "/app/pp_output")
log_dir = os.getenv("LOG_DIR", "/app/logs")

time_series_data = {"time": [], "output_files": [], "pp_output_files": []}


# Function to count the number of files in a directory
def count_files_in_directory(directory):
    try:
        path = Path(directory)
        return len(list(path.glob("*"))) 
    except Exception as e:
        return 0 


def read_last_log_entries(log_file, lines=5):
    try:
        with open(log_file, "r") as file:
            return "".join(file.readlines()[-lines:])
    except FileNotFoundError:
        return "No log file found."


def calculate_differences(series):
    if len(series) < 2:
        return [0] 
    return [series[i] - series[i - 1] for i in range(1, len(series))]


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = html.Div(
    [
        html.H1(
            "File Processing Progress Dashboard",
            style={"textAlign": "center", "margin-bottom": "30px"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Input Files", style={"textAlign": "center"}),
                        html.Div(
                            id="input-file-count",
                            style={"fontSize": "24px", "textAlign": "center"},
                        ),
                        dbc.Progress(
                            id="input-progress-bar",
                            striped=True,
                            animated=True,
                            color="primary",
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.H3("Output Files", style={"textAlign": "center"}),
                        html.Div(
                            id="output-file-count",
                            style={"fontSize": "24px", "textAlign": "center"},
                        ),
                        dbc.Progress(
                            id="output-progress-bar",
                            striped=True,
                            animated=True,
                            color="success",
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.H3("Post-Processed Files", style={"textAlign": "center"}),
                        html.Div(
                            id="pp-output-file-count",
                            style={"fontSize": "24px", "textAlign": "center"},
                        ),
                        dbc.Progress(
                            id="pp-output-progress-bar",
                            striped=True,
                            animated=True,
                            color="info",
                        ),
                    ],
                    width=4,
                ),
            ],
            style={"margin-bottom": "30px"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Raw Consumer Logs"),
                        html.Pre(
                            id="raw-log",
                            style={
                                "whiteSpace": "pre-wrap",
                                "wordBreak": "break-word",
                                "backgroundColor": "#f8f9fa",
                                "padding": "10px",
                            },
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.H4("Preprocessing Logs"),
                        html.Pre(
                            id="preprocessing-log",
                            style={
                                "whiteSpace": "pre-wrap",
                                "wordBreak": "break-word",
                                "backgroundColor": "#f8f9fa",
                                "padding": "10px",
                            },
                        ),
                    ],
                    width=6,
                ),
            ],
            style={"margin-bottom": "30px"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id="time-series-graph",
                            config={"displayModeBar": False},
                            style={"height": "400px"},
                        )
                    ]
                )
            ]
        ),
        dcc.Interval(
            id="interval-component",
            interval=10 * 1000,  # Update every 10 seconds
            n_intervals=0,
        ),
    ]
)


@app.callback(
    [
        Output("input-file-count", "children"),
        Output("output-file-count", "children"),
        Output("pp-output-file-count", "children"),
        Output("input-progress-bar", "value"),
        Output("output-progress-bar", "value"),
        Output("pp-output-progress-bar", "value"),
        Output("input-progress-bar", "label"),
        Output("output-progress-bar", "label"),
        Output("pp-output-progress-bar", "label"),
        Output("raw-log", "children"),
        Output("preprocessing-log", "children"),
        Output("time-series-graph", "figure"),
    ],
    [Input("interval-component", "n_intervals")],
)
def update_dashboard(n):
    # File counts
    input_count = count_files_in_directory(dir_input)
    output_count = count_files_in_directory(dir_output)
    pp_output_count = count_files_in_directory(dir_pp_output)

    total_files = max(
        input_count, output_count, pp_output_count
    ) 
    if total_files == 0:
        total_files = 1 

    input_progress = int((input_count / total_files) * 100)
    output_progress = int((output_count / total_files) * 100)
    pp_output_progress = int((pp_output_count / total_files) * 100)

    raw_log = read_last_log_entries(f"{log_dir}/raw/raw.log", lines=5)
    preprocessing_log = read_last_log_entries(
        f"{log_dir}/preprocessing/preprocessing.log", lines=5
    )

    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    time_series_data["time"].append(current_time)
    time_series_data["output_files"].append(output_count)
    time_series_data["pp_output_files"].append(pp_output_count)

    output_diff = calculate_differences(time_series_data["output_files"])
    pp_output_diff = calculate_differences(time_series_data["pp_output_files"])

    time_series_figure = {
        "data": [
            {
                "x": time_series_data["time"][1:],
                "y": output_diff,
                "type": "line",
                "name": "Output Files",
                "line": {"color": "green"},
            },
            {
                "x": time_series_data["time"][1:],
                "y": pp_output_diff,
                "type": "line",
                "name": "Post-Processed Files",
                "line": {"color": "blue", "dash": "dash"},
            },
        ],
        "layout": {
            "title": "Change in File Counts Over Time",
            "xaxis": {"title": "Time"},
            "yaxis": {
                "title": "Change in File Count",
                "rangemode": "tozero",
                "automargin": True,
            },
            "showlegend": True,
            "height": 400,
            "margin": {"l": 50, "r": 30, "t": 50, "b": 40},
            "plot_bgcolor": "#f8f9fa",
        },
    }

    return (
        f"Total Input Files: {input_count}",
        f"Total Output Files: {output_count}",
        f"Total Post-Processed Files: {pp_output_count}",
        input_progress,
        output_progress,
        pp_output_progress,
        f"{input_progress}%",
        f"{output_progress}%",
        f"{pp_output_progress}%",
        raw_log,
        preprocessing_log,
        time_series_figure,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
