#test metrics (beyond certain treshold)

#prediction on known input 

#predictions only of kind Yes or No 

import sys ,pandas as pd,joblib
sys.path.append("../")

import requests,pandas as pd,numpy as np
from oct24_bmlops_int_weather.src.utils import *

model = joblib.load(f"{model_path}")

test_df = pd.read_csv(f"{ref_path}",index_col=0)

print(model)