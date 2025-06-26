import sys ,pandas as pd
sys.path.append("../")

import requests,pandas as pd,numpy as np
from steps.utils import *
import joblib


input_test = {'Location_encoded': 0.198930, 'MinTemp': 17.0, 'MaxTemp': 25.9, 'Rainfall': 0.0, 
              'Evaporation': 9.0, 'Sunshine': 11.3, 'WindGustDir': 'SW', 'WindGustSpeed': 54.0, 
              'WindDir9am': 'SSE', 'WindDir3pm': 'SSW', 'WindSpeed9am': 17.0, 'WindSpeed3pm': 26.0,
                'Humidity9am': 59.0, 'Humidity3pm': 46.0, 'Pressure9am': 1017.9, 'Pressure3pm': 1015.2,
                  'Cloud9am': 6.0, 'Cloud3pm': 1.0, 'Temp9am': 19.1, 'Temp3pm': 24.4, 'RainToday': 'No', 
           
                   }


response = requests.post("http://localhost:8000/predict/6", json=input_test)
print(response.text)