# import functions to preprocess data and csv file
#import function to split data to x and y

# import grid searched params or best model

# predict incoming data with preprocessing pipeline and produce output

import os,sys
import pandas as pd
sys.path.append("../")
import site
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score,matthews_corrcoef
#from models import random
import mlflow,yaml
from pathlib import Path 
from site import addsitedir 
from train_model import Trainer
from dataclasses import dataclass,field
from utils import *


#sys.path.append(str(Path(__file__).parent)/ 'src')


@dataclass
class Predictor(Trainer):
    def __init__(self):

        self.model_path = os.path.join(home_dir,'steps/model.pkl')
        self.model = self.load_model()
        
        
    def load_model(self):
        
        model = joblib.load(self.model_path)
        return model
    

    def feature_target_separator(self, data):
        X = data.drop("RainTomorrow",axis=1)
        y = data["RainTomorrow"]
        return X, y

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        #accuracy = accuracy_score(y_test, y_pred)
        #class_report = classification_report(y_test, y_pred)
        matthews_score = matthews_corrcoef(y_test, y_pred)
        return matthews_score
    



