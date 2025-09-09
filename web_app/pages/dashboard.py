import dash
from dash import html, dcc
import plotly.express as px
from starlette.middleware.wsgi import WSGIMiddleware


from inference_pipeline import predict

import pandas as pd

dash.register_page(
    __name__,
    path="/analytics",
    name="Analytics",
    title="Analytics — Dash App",
    description="Simple analytics demo with a Plotly graph."
)

cities_data  = {
    'City': [
        'Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Darwin'
    ],
    'Latitude': [
        -33.865143,  # Sydney
        -37.840935,  # Melbourne
        -27.470125,  # Brisbane
        -31.953512,  # Perth
        -34.921230,  # Adelaide
        -12.462827   # Darwin
    ],
    'Longitude': [
        151.209900,  # Sydney
        144.946457,  # Melbourne
        153.021072,  # Brisbane
        115.857048,  # Perth
        138.599503,  # Adelaide
        130.841782   # Darwin
    ],

    'RainProbability': [predict(i) for i in ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Darwin'] ]  # Replace with real predictions
}

df = pd.DataFrame(cities_data)

# Create map figure




fig = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    hover_name="City",
    hover_data={"RainProbability": ':.2/100% percent', "Latitude": True, "Longitude": True},
    color="RainProbability",
    color_continuous_scale=px.colors.sequential.Blues,
    size="RainProbability",
    size_max=20,
    zoom=3,
    center={"lat": -25, "lon": 135}
)

fig.add_scattermapbox(
        lat=[cities_data["Latitude"] for d in cities_data],
        lon=[cities_data["Longitude"] for d in cities_data],
        mode="markers",
        marker=dict(size=6, color="black"),
        hoverinfo="none",
        showlegend=False
    )

fig.update_layout(
        mapbox=dict(
            style = 'carto-positron',
            #style="open-street-map",
              # Rotate slightly
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        font=dict(family="Lato, sans-serif")
    )
# CSS raindrop animation
rain_css =  """
<style>
.rain-container {
  position: absolute;
  width: 100%;
  height: 100%;
  pointer-events: none; /* allow clicking through to the map */
  overflow: hidden;
}

.drop {
  position: absolute;
  width: 2px;
  height: 10px;
  background: rgba(0,0,255,0.5);
  bottom: 100%;
  animation: fall 0.8s linear infinite;
}

@keyframes fall {
  to {
    transform: translateY(100vh);
  }
}
</style>
 """

# Generate raindrops for cities with >60% rain probability
def generate_rain_effects(df, threshold=0.6, drops=50):
    rain_effects = []
    for _, row in df.iterrows():
        if row["RainProbability"] > threshold:
            rain_effects.append(
                html.Div(
                    className="rain-container",
                    children=[
                        html.Div(className="drop", style={"left": f"{i*2}%"})
                        for i in range(drops)
                    ]
                )
            )
    return rain_effects

def latlon_to_position(lat, lon):
    # Approximate bounding box of Australia for normalization
    lat_min, lat_max = -44, -10   # South to North
    lon_min, lon_max = 113, 154   # West to East

    top = (lat_max - lat) / (lat_max - lat_min) * 100  # invert lat (higher = closer to top)
    left = (lon - lon_min) / (lon_max - lon_min) * 100

    return f"{top}%", f"{left}%"

def generate_city_rain(df, threshold=0.6, drops=30):
    rain_effects = []
    for _, row in df.iterrows():
        if row["RainProbability"] > threshold:
            top, left = latlon_to_position(row["Latitude"], row["Longitude"])
            rain_effects.append(
                html.Div(
                    className="rain-container",
                    style={"top": top, "left": left, "width": "80px", "height": "120px"},
                    children=[
                        html.Div(className="drop", style={"left": f"{i*3}%"})
                        for i in range(drops)
                    ]
                )
            )
    return rain_effects


layout = html.Div([
    html.H2("Forecasted Next Day Rain Probability in Major Australian Cities"),
    dcc.Graph(figure=fig, style={"height": "80vh"}),
    html.Div(generate_rain_effects(df)),
    html.Div(dcc.Markdown(rain_css), style={"display": "none"})  # inject CSS
])
# ---------- Create FastAPI App and Mount Dash ----------



