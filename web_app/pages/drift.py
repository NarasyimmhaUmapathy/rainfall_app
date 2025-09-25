import sys,os
sys.path.append('../../')

import pandas as pd
import datetime
import dash
from dash import dash_table
from dash import html, callback, Input, Output
import glob

from monitor import feature_metrics
from utils import home_dir

REPORT_DIR = os.path.abspath(os.path.join(home_dir, "reports"))



dash.register_page(__name__)

def get_report_data(report_dir="reports"):
    files = sorted(glob.glob(os.path.join(REPORT_DIR, "*.html")), reverse=True)
    data = []
    for f in files:
        filename = os.path.basename(f)
        created = pd.to_datetime(os.path.getmtime(f), unit="s").strftime("%Y-%m-%d %H:%M:%S")
        link = f"[Open](/reports/{filename})"  # markdown link
        data.append({"Report Name": filename, "Created": created, "Link": link})
    return pd.DataFrame(data)



# Page layout
layout = html.Div(
    className="page page-drift",
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "2rem",
        "fontFamily": "Arial, sans-serif",
    },
    children=[
        html.H1("📊 Drift Monitoring Dashboard", style={"textAlign": "center", "marginBottom": "1rem"}),

        html.Div(
            style={"textAlign": "center", "marginBottom": "2rem"},
            children=[
                html.Button(
                    "⚡ Generate New Report",
                    id="refresh-btn",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#4CAF50",
                        "color": "white",
                        "padding": "10px 20px",
                        "border": "none",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                        "fontSize": "16px",
                    }
                ),
            ]
        ),

        html.Div(
            children=[
                html.Iframe(
                    id="report-frame",
                    style={
                        "width": "100%",
                        "height": "600px",
                        "border": "2px solid #ddd",
                        "borderRadius": "10px",
                        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
                    }
                )
            ]
        ),

        html.H2("📂 Previous Reports", style={"marginTop": "2rem", "marginBottom": "1rem"}),

         dash_table.DataTable(
            id="report-table",
            columns=[
                {"name": "Report Name", "id": "Report Name"},
                {"name": "Created", "id": "Created"},
                {"name": "Link", "id": "Link", "presentation": "markdown"},
            ],
            data=get_report_data().to_dict("records"),
            filter_action="native",   # 🔎 search bar
            sort_action="native",     # ↕️ sortable
            page_size=10,             # 📜 pagination
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px"},
            style_header={"fontWeight": "bold", "borderBottom": "2px solid #ddd"},
         )
            ]
        )

@callback(
    Output("report-frame", "src"),
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call=False
)
def update_report(n_clicks):
    report_path, _ = feature_metrics(1)
    filename = os.path.basename(report_path)
    return f"reports/{filename}"

    




""" 
def list_reports():
    files = sorted(glob.glob(f"{home_dir}/reports/*.html"), reverse=True)
    links = [html.Li(html.A(os.path.basename(f), href=f"/reports/{os.path.basename(f)}", target="_blank")) for f in files]
    return html.Ul(links)

dash.register_page(
    __name__,
    path="/drift",
    name="Drift Report",
    title="Model Drift Monitoring",
    description="Generated drift monitoring report."
)

layout = html.Div(
    className="page page-drift",
    children=[
        html.H1("Drift Report"),
        html.Button("Generate New Report", id="refresh-btn", n_clicks=0, className="btn"),
        html.Iframe(
            id="report-frame",
            style={"width": "100%", "height": "600px", "border": "none", "marginTop": "20px"}
        ),
        html.H2("Previous Reports"),
        html.Div(id="report-list", children=list_reports())
    ]
)

@callback(
    Output("report-frame", "src"),
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call=False
)
def update_report(n_clicks):
    report_path, _ = feature_metrics(1)
    filename = os.path.basename(report_path)
    return f"reports/{filename}"

 """

