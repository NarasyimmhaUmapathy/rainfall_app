import sys,os
sys.path.append('../')

import dash
from dash import html,dcc

from utils import reports_path,home_dir
from flask import send_from_directory


def Navbar():
    # Build links automatically from the page registry, keeping Home first
    pages = list(dash.page_registry.values())
    pages_sorted = sorted(pages, key=lambda p: 0 if p["path"] == "/" else 1)

    links = []
    for p in pages_sorted:
        if p.get("is_hidden"):
            continue
        links.append(
            dcc.Link(
                p.get("name", p["title"]),
                href=p["path"],
                className="nav-link",
                style={
                    "margin": "0 12px",
                    "textDecoration": "none",
                    "color": "#f0f0f0",
                    "fontWeight": "500",
                }
            )
        )

    return html.Nav(
        className="navbar",
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "padding": "0.8rem 2rem",
            "backgroundColor": "#2c3e50",
            "color": "white",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.2)",
            "position": "sticky",
            "top": "0",
            "zIndex": "1000",
        },
        children=[
            html.Div("🚀 Dash Multi-Page", className="brand", style={
                "fontWeight": "700",
                "fontSize": "1.2rem",
                "letterSpacing": "0.5px"
            }),
            html.Div(links, className="nav-links", style={"display": "flex"})
        ]
    )

# Initialize app with Dash Pages
app = dash.Dash(
    __name__,
    use_pages=True,  
                             # enables the multi-page router
    suppress_callback_exceptions=True,  # lets pages register callbacks safely
    title="My Multi-Page Dash App",     # default <title>, pages can override
    update_title=None                   # no "Updating..." in the tab
)

REPORT_DIR = os.path.abspath(os.path.join(home_dir, "reports"))

@app.server.route(f"/reports/<path:filename>")
def download_report(filename):
    return send_from_directory(f"{REPORT_DIR}", filename)

# Global layout: Navbar + the current page
app.layout = html.Div(
    children=[
        Navbar(),
        html.Div(
            dash.page_container,
            style={
                "padding": "2rem",
                "maxWidth": "1200px",
                "margin": "0 auto",
                "fontFamily": "Lato, sans-serif",
            }
        )
    ],
    className="app-shell",
    style={
        "backgroundColor": "#f7f9fb",
        "minHeight": "100vh"
    }
)
server = app.server  # for WSGI/production

#if __name__ == "__main__":
    #app.run_server(debug=True)
    #app.run(host='127.0.0.1', port=8050)
