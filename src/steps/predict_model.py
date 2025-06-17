# import functions to preprocess data and csv file
#import function to split data to x and y

# import grid searched params or best model

# predict incoming data with preprocessing pipeline and produce output

import os,sys,logging,numpy as np
from sklearn.base import clone
import pandas as pd
sys.path.append("../")
import site
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score,matthews_corrcoef
#from models import random
import mlflow,yaml
from pathlib import Path 
import datetime,json
from site import addsitedir 
from steps.train_model import Trainer,Utils
from dataclasses import dataclass,field
from utils import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold,cross_validate
from sklearn.ensemble import AdaBoostClassifier
from functools import lru_cache
from utils import *
from sklearn.metrics import fbeta_score, make_scorer,matthews_corrcoef,f1_score,average_precision_score,recall_score
from mlflow.models import infer_signature



matthews_scores = make_scorer(matthews_corrcoef)
f1_scores = make_scorer(f1_score)
avg_prec_metric = make_scorer(average_precision_score)
recall_metric = make_scorer(recall_score)

#sys.path.append(str(Path(__file__).parent)/ 'src')

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{home_dir}/logs/prediction_pipeline.log', encoding='utf-8', level=logging.INFO,format='%(asctime)s %(message)s')


@dataclass
class Model(Utils):
    def __init__(self):
        super().__init__()
        self.saved_model = self.config["model"]["saved_model"]
        self.registered_model = self.config["model"]["registered_model"]


    def return_model(self) -> Pipeline:
        model = joblib.load(f'{home_dir}/models/{self.saved_model}')
        logging.info(f"loading model with name {self.saved_model} pipeline from models dir")
        assert type(model) == Pipeline
        return model

    
    def update_model(self,model,train_dir,prod_dir):
        X_train,y_train = self.load_data(train_dir)
        X_prod,y_prod = self.load_data(prod_dir)
       # model = mlflow.sklearn.load_model(f"models:/test_monitoring/Staging")
        model.fit(X_prod,y_prod)
        logging.info("updating local version of model in models dir from predict.py module")

        joblib.dump(model,f"./models/{self.saved_model}.pkl")
        logging.info("updating model version in mlflow from predict.py module")

        mlflow.sklearn.log_model(
      sk_model=model,
      input_example=X_train.fillna(),
      artifact_path="models",
      registered_model_name=self.registered_model
        )



       
       

    def update_params(self,params,model_path:str,name:str)-> str:
        X,y = self.load_data(dir)

        model = joblib.load(f'{home_dir}/models/{self.model_name}')
        model.named_steps["model"].set_params(**params) 

        signature = infer_signature(X, model.predict(X))

    #insert logging step    

        model_info=mlflow.sklearn.log_model(
      sk_model=model,
      artifact_path=f'{home_dir}/models/{self.model_name}',
      input_example=X,
      registered_model_name=name
        )

        return model_info.model_uri
    
    

    def evaluate_model(self,model,dir,num_splits):
        X_test,y_test = self.load_data(dir)
        sfold = StratifiedKFold(n_splits=num_splits,random_state=self.random_state,shuffle=True)
        metrics = {"precision_score": avg_prec_metric, "recall_score": recall_metric,"f1_score":f1_scores}
        cvs = cross_validate(model,X_test,y_test,cv=sfold,scoring=metrics,n_jobs=-1)
        #insert stratified cross validate func and log the mean scores.
        #accuracy = accuracy_score(y_test, y_pred)
        #class_report = classification_report(y_test, y_pred)
        #scores = [cvs["test_precision_score"],cvs["test_recall_score"],cvs["test_f1_score"]]
        
        return cvs
    

    

   
   



    