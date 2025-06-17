import joblib
import sys
sys.path.append("../")

from fastapi import FastAPI
import numpy as np
from monitoring.scripts import return_mapping,feature_metrics,model_metrics,ref_data_test,prod_data

from starlette.requests import Request
from utils import model_path
from dataclasses import dataclass,astuple
import logging
from typing import Callable, Text

from evidently import ColumnMapping
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    Response,
    FileResponse
)
from evidently._pydantic_compat import BaseModel
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='FASTAPI_APP - %(asctime)s - %(levelname)s - %(message)s'
)


app = FastAPI()
model = joblib.load(f"{model_path}")

column_mapping = return_mapping("target","prediction")

@dataclass
class Inference(BaseModel):

    Location:str 
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
    month : float 
    day: float 
    year: float

    def __iter__(self):
        return iter(astuple(self))

@app.get('/hello')
def hello():
    return {"hello there!"}



@app.post('/predict')
def predict(payload:Inference):

   

    input_elems = [[payload.Location,payload.WindGustDir,
                        payload.WindDir9am,payload.WindDir3pm,payload.RainToday,payload.MinTemp,
                   payload.MaxTemp,payload.Rainfall,payload.Evaporation,payload.Sunshine,
                        payload.WindGustSpeed,payload.WindSpeed9am,payload.WindSpeed3pm,payload.Humidity9am,
                        payload.Humidity3pm,payload.Pressure9am,payload.Pressure3pm,payload.Cloud9am,payload.Cloud3pm,
                        payload.Temp9am,payload.Temp3pm,payload.month,payload.day,payload.year]]
                        
    prediction = model.predict(input_elems)

    return {f"predicted rain chance for tommorow is {prediction}"}



@app.get('/monitor-feature-drift')
def monitor_model_performance(window_size: int = 14) -> FileResponse:

    logging.info('Read current data')
    current_data: pd.DataFrame = prod_data

    logging.info('Read reference data')
    reference_data = ref_data_test

    logging.info('Build report')
    column_mapping: ColumnMapping = column_mapping
    report_path: Text = feature_metrics(
       window_size,1
    )

    logging.info('Return report as html')
    return FileResponse(report_path)


