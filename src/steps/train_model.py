import sys,os
sys.path.append('../')
import joblib
import yaml
import mlflow
from src.steps.ingest_data import *

from sklearn.preprocessing import TargetEncoder, RobustScaler,OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score,matthews_corrcoef,roc_auc_score
import pandas as pd
import category_encoders as ce
import optuna
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import time
from dataclasses import dataclass,field
from abc import ABC,abstractmethod
from src.steps.utils import *

class Utils():

    def __init__(self):
          self.home_dir = self.load_config["directories"]["home"]
          self.train_dir = self.load_config["directories"]["train"]
          self.test_dir = self.load_config["directories"]["train"]
          self.model_params = self.load_config["model"]["params"]
          self.model_name = self.load_config["model"]["model_name"]
          self.challenger_name = self.load_config["model"]["challenger_name"]

          self.mlflow_name = self.load_config["model"]["mlflow_name"]

          self.model_path = self.load_config['model']['store_path']
          self.categorical_features = self.load_config['cat_features']
          self.numerical_features = self.load_config['num_features']
          self.target = self.load_config['target']
          self.test_ratio = self.load_config["train_test_ratio"]
          self.random_state = self.load_config["random_state"]

    @classmethod
    def load_config(self):
        with open(conf_path, 'r') as config_file:
            return yaml.safe_load(config_file.read())

    @classmethod
    def feature_target_separator(self,data):
        X = data.drop(self.config["target"],axis=1)
        y = data[self.config["target"]]
        return X, y
    
 

  


@dataclass
class Trainer(Utils):
    def __init__(self):
        self.config = load_config()
       
        self.home_dir = home_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model_params = self.config["model"]["params"]
        self.model_name =  self.config["model"]["model_name"]
        self.challenger_name =  self.config["model"]["challenger_name"]

        self.mlflow_name = self.config["model"]["mlflow_name"]

        self.model_path = self.config['model']['store_path']
        self.categorical_features = self.config['cat_features']
        self.numerical_features = self.config['num_features']
        self.target = self.config['target']
        self.test_ratio = self.config["train_test_ratio"]
        self.random_state = self.config["random_state"]
  

        pipeline = field(default_factory=self.create_pipeline())
        
 

    def create_pipeline(self):
        processor = ColumnTransformer(     
            transformers=[  
       # ("impute_categorical",SimpleImputer(strategy="most_frequent"),self.categorical_features),
        ('scale',RobustScaler(),self.numerical_features),
        ("impute_numerical",SimpleImputer(),self.numerical_features),
        ("encoding_categorical_features",TargetEncoder(),self.categorical_features)
            ]
        )

        model_map = {
            'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
            'AdaBoostClassifier':AdaBoostClassifier
        }
    
        model_class = model_map[self.model_name]
        model = model_class(**self.model_params)


        pipeline = Pipeline([
        ('processor',processor),
        ('model',model)
        ])
        
 

        return pipeline
    
    def load_data(self):


        home_string  = " ".join(str(x) for x in self.home_dir)
        data_string =  " ".join(str(x) for x in self.train_dir)


        data = pd.read_csv(f'{home_string}/{data_string}',index_col=0)

        X,y = self.feature_target_separator(data)

        return X,y



    def feature_target_separator(self, data):
        X = data.drop(self.config["target"],axis=1)
        y = data[self.config["target"]]
        return X, y
    
    
      

    def save_model(self):
        model_file_path = os.path.join(self.model_path,f'{self.model_name}.pkl')
        joblib.dump(self.pipeline, model_file_path)





def get_or_create_experiment(experiment_name:str):

        if experiment := mlflow.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        else:
   
            return mlflow.create_experiment(experiment_name)




@dataclass
class MLflow(Trainer):

    def __init__(self):
        self.config = load_config()
        self.model_name = self.config["model"]["model_name"]
        trainer = field(default=Trainer())
        self.champion_name = self.config["model"]["mlflow_name"]


    def tuning(self,objective,experiment_id,run_name):


        with mlflow.start_run(experiment_id=experiment_id,run_name=run_name,nested=True):
  # Initialize the Optuna study
            study = optuna.create_study(directions=['maximize', 'maximize', 'maximize', 'maximize'])

  # Execute the hyperparameter optimization trials.
  # Note the addition of the `champion_callback` inclusion to control our logging
            study.optimize(objective, n_trials=50)

            mlflow.log_params(study.best_params)
            mlflow.log_metric("best f1 and maxwell coeff score", study.best_value)

  # Log tags
            mlflow.set_tags(
        tags={
          "project": "australia weather forecasting",
          "optimizer_engine": "optuna",
          "model_family": "sklearn",
          "feature_set_version": 1,
            }
    	        )
        
        best_params = study.best_params
        best_metrics = study.best_value
        

        return best_params,best_metrics
    

    def save_model(self,params,model_path:str,name:str)-> str:


        model = self.create_pipeline()
        X,y = self.load_data()
        model.named_steps["model"].set_params(**params) 
        model.fit(X,y)

        signature = infer_signature(X, model.predict(X))

    #insert logging step    

        model_info=mlflow.sklearn.log_model(
      sk_model=model,
      artifact_path=model_path,
      input_example=X,
      registered_model_name=name
        )

        return model_info.model_uri
    

    
