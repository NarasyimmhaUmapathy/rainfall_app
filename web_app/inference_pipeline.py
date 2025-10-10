import joblib
import sys,pickle
import numpy as np
from shapash import SmartExplainer

sys.path.append("../")

from fastapi import FastAPI
import dash
from dash import dcc, html

from monitoring.monitor import return_mapping,feature_metrics

import plotly.express as px
from starlette.middleware.wsgi import WSGIMiddleware



from starlette.requests import Request
from src.steps.utils import *
from dataclasses import dataclass,astuple
import logging
from typing import Callable, Text

from evidently import ColumnMapping
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager


external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap"
]


from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    Response,
    FileResponse
)
import pandas as pd
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='FASTAPI_APP - %(asctime)s - %(levelname)s - %(message)s'
)



app = FastAPI()


dash_app = dash.Dash(
    __name__,
    requests_pathname_prefix="/australia_rainfall_chance/"
)




preprocessor = joblib.load(f"{preprocessor_path}")
prod_data = pd.read_csv(f"{prod_path}",index_col=0)
train_data = pd.read_csv(f"{train_path}",index_col=0)

X_prod = prod_data.drop("Location",axis=1)
y_prod = prod_data["Location"]
model = joblib.load(model_path)
xpl = SmartExplainer(model=model)







@dataclass
class Inference(BaseModel):

    Location_encoded:float 
    Cloud9am : float
    Cloud3pm : float
    WindGustDir:str
    WindDir9am : str 
    RainToday: str 
    MinTemp : float
    MaxTemp : float 
    Temp9am : float 
    Temp3pm : float
    Rainfall : float 
    Evaporation : float 
    Sunshine : float 
    WindGustSpeed : float 
    WindDir3pm : str
    WindSpeed9am : float 
    WindSpeed3pm : float 
    Humidity9am : float 
    Humidity3pm : float 
    Pressure9am : float 
    Pressure3pm : float 
   

    def __iter__(self):
        return iter(astuple(self))
    



@app.get('/predict/{location}')
def predict(location:str):

    model = joblib.load(f"{model_path}")
 

    preprocessor = joblib.load(preprocessor_path)
    
    # keep transformed features in prod data
    # just load the model and perform prediction on filtered prod data
    # compute prediction for user given location
    data = prod_data[prod_data["Location"] == location]
    
    data = data[data["Date"] == data["Date"].max()]

    data = data.drop(["Location","Date","day"],axis=1)
    test = preprocessor.transform(data)


    preds = model.predict_proba(test)
    #output days from now 

   # return (f"tommorow's rain forecast for the location {location} days is {preds[0][0]*100} percent")
    return round(preds[0][0]*100,2)

    # return preds
    

    

@app.get('/drift_monitoring/{window_size}')
def monitor_model_performance(window_size: int = 1) -> FileResponse:

  

    logging.info(f'Read reference data from {ref_path}')
    reference_data = pd.read_csv(f"{ref_path}")

    current_data: pd.DataFrame = reference_data.query("Date >= '2017-03-26' \
                       and Date <= '2017-06-26'")

    logging.info('Build report')
    column_mapping: ColumnMapping = return_mapping("target","prediction")
    report_path: Text = feature_metrics(
       window_size
    )[0]

    logging.info('Return report as html')
    return FileResponse(report_path)
    


# Sample data: replace with your ML model's predictions
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
            style="open-street-map",
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


dash_app.layout = html.Div([
    html.H2("Forecasted Next Day Rain Probability in Major Australian Cities"),
    dcc.Graph(figure=fig, style={"height": "80vh"}),
    html.Div(generate_rain_effects(df)),
    html.Div(dcc.Markdown(rain_css), style={"display": "none"})  # inject CSS
])
# ---------- Create FastAPI App and Mount Dash ----------


app.mount("/australia_rainfall_chance", WSGIMiddleware(dash_app.server))