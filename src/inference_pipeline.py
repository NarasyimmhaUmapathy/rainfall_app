import joblib
import sys
sys.path.append("../")

from fastapi import FastAPI
import numpy as np
from monitor import return_mapping,feature_metrics

from starlette.requests import Request
from src.steps.utils import model_path,ref_path,load_config,preprocessor_path,prod_processed_path,home_dir
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
import pandas as pd
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='FASTAPI_APP - %(asctime)s - %(levelname)s - %(message)s'
)


app = FastAPI()
model = joblib.load(f"{model_path}")
preprocessor = joblib.load(f"{preprocessor_path}")
prod_data = pd.read_csv(f"{home_dir}/src/output_data/prod/prediction_production_data.csv",index_col=0)








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
    



@app.get('/hello')
def hello():
    return {"hello there!"}



@app.post('/predict/{num_days}')
def predict(num_days:int):

    
    # keep transformed features in prod data
    # just load the model and perform prediction on filtered prod data
    preds = model.predict(prod_data[:num_days])

    #output days from now 

    return (f"rain forecast for the next {num_days} days is {preds}")

    # return preds
    

    



@app.get('/monitor-feature-drift/{window_size}')
def monitor_model_performance(window_size: int = 1) -> FileResponse:

  

    logging.info(f'Read reference data from {ref_path}')
    reference_data = pd.read_csv(f"{ref_path}")

    logging.info(f'Read current data from {ref_path}') #change to separate prod dir
    current_data: pd.DataFrame = reference_data.query("Date >= '2017-03-26' \
                       and Date <= '2017-06-26'")

    logging.info('Build report')
    column_mapping: ColumnMapping = return_mapping("target","prediction")
    report_path: Text = feature_metrics(
       window_size
    )

    logging.info('Return report as html')
    return FileResponse(report_path)


