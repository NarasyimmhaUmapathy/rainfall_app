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
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
#from models import random
import mlflow,yaml
from pathlib import Path 
from datetime import datetime
from steps.train_model import Utils
import json
from site import addsitedir 
from dataclasses import dataclass,field
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold,cross_validate
from sklearn.ensemble import AdaBoostClassifier
from functools import lru_cache
from steps.utils import *
from sklearn.metrics import fbeta_score, make_scorer,matthews_corrcoef,f1_score,average_precision_score,recall_score
from mlflow.models import infer_signature


f1_scores = make_scorer(f1_score)
recall_scores = make_scorer(recall_score)


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

    def update_model(self,model,prod_dir):

        assert prod_dir != train_path

        date_time = datetime.today().strftime('%Y-%m-%d')
        prod_data = pd.read_csv(f"{prod_dir}",index_col=0)
        X_prod = prod_data.drop("Target_encoded",axis=1)
        y_prod = prod_data["Target_encoded"]

        

        preprocessor = joblib.load(f"{preprocessor_path}")


        preprocessor.fit(X_prod,y_prod)

        print("data fitted with preprocessor")


        model.fit(preprocessor.transform(X_prod),y_prod)

        print("model updated with prod data")

        
        logging.info("updating local version of model in models dir from predict.py module")

        joblib.dump(model,f"./models/{self.saved_model}_{date_time}.pkl")
        logging.info("updating model version in mlflow from predict.py module")

        mlflow.sklearn.log_model(
      sk_model=model,
      input_example=preprocessor.transform(X_prod),
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
    
    
    def evaluate_model(self,model,data_path,cv_splits:int):

        X_test,y_test = self.load_data(data_path)

        

        sfold = StratifiedKFold(cv_splits,random_state=self.random_state,shuffle=True)
        metrics = {"f1_score":f1_scores,"recall_score":recall_scores}

        preprocessor = joblib.load(f"{preprocessor_path}")
        conf = load_config()
        cols = conf["num_features"] + conf["cat_features"] 
        cvs = cross_validate(model,preprocessor.transform(X_test[cols])
                     ,y_test,cv=sfold,scoring=metrics,n_jobs=-1)
        
        preds = model.predict(preprocessor.transform(X_test[cols]))
        ConfusionMatrixDisplay.from_predictions(
        y_test, preds)

        date_time = datetime.today().strftime('%Y-%m-%d')

        plt.savefig(f"{home_dir}/reports/confusion_matrix_{date_time}.png")


      
        return cvs["test_f1_score"].mean(),cvs["test_recall_score"].mean()
    

    

   
   



    