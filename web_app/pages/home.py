
import dash
from dash import html, dcc, callback, Input, Output

dash.register_page(
    __name__,
    path="/",
    name="Home",
    title="Welcome — Dash App",
    description="Landing page for the multi-page Dash example."
)

layout = html.Div(
    className="page page-home",
    children=[
        html.Div(
            className="hero",
            children=[
                html.H1("Welcome 👋"),
                html.P(
                    "This is your landing page. Use the navbar to explore other pages."
                ),
                dcc.Link("Go to Analytics →", href="/analytics", className="btn")
            ]
        ),
        html.Div(className="card-grid", children=[
            html.Div(className="card", children=[
                html.H3("Fast routing"),
                html.P("Dash Pages handles URLs and titles automatically.")
            ]),
            html.Div(className="card", children=[
                html.H3("Composable UI"),
                html.P("Keep shared UI (like navbars) outside page layouts.")
            ]),
            html.Div(className="card", children=[
                html.H3("Callback-ready"),
                html.P("Each page can ship its own callbacks and state.")
            ]),
        ]),
        html.Div(className="cta", children=[
            html.Label("Try a quick interaction:"),
            dcc.Input(id="name-input", placeholder="Type your name…", className="input"),
            html.Div(id="greet-out", className="muted")
        ])
    ]
)

@callback(Output("greet-out", "children"), Input("name-input", "value"))
def greet(name):
    if not name:
        return "Tip: your changes reflect instantly without reloading."
    return f"Nice to meet you, {name}!"
