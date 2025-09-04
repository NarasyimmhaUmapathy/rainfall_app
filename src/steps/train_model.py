import sys,os
sys.path.append('../')
import joblib
import yaml
import mlflow
from steps.ingest_data import *

from sklearn.preprocessing import TargetEncoder, RobustScaler,OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold,cross_validate

from sklearn.metrics import f1_score,matthews_corrcoef,roc_auc_score,make_scorer,recall_score
import pandas as pd
#import optuna
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import time
from dataclasses import dataclass,field
from abc import ABC,abstractmethod
from steps.utils import *
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"{home_dir}/logs/training_pipeline.log", filemode="a",encoding='utf-8', level=logging.INFO,
                    format='%(asctime)s %(filename)s->%(funcName)s():%(lineno)s %(message)s ')






class Utils():

    def __init__(self):
          self.config = self.load_config()
          self.home_dir = home_dir
          self.train_dir = train_path
          self.test_dir = test_path
          self.model_path = model_path
          self.preprocessor_path = preprocessor_path
       
          self.categorical_features = self.config['cat_features']
          self.numerical_features = self.config['num_features']
          self.target = self.config['target_encoded']
          self.model_features = self.numerical_features + self.categorical_features + self.target
          self.test_ratio = self.config["train_test_ratio"]
          self.random_state = self.config["random_state"]


    def load_config(self):
        with open(conf_path, 'r') as config_file:
            return yaml.safe_load(config_file.read())

    def feature_target_separator(self,data):

        if type(data) == np.ndarray:
            data = pd.DataFrame(data,columns=self.model_features)
        X = data.drop(self.target,axis=1)
        y = data[self.target]
        return X,y

    @staticmethod
    def return_metrics():
       
        f1_metric = make_scorer(f1_score)
        recall_metric = make_scorer(recall_score)

        return f1_metric,recall_metric
    
    def load_data(self,path):

        #home_string  = "".join(str(x) for x in home_dir)
        #data_string =  "".join(str(x) for x in train_dir)

        #logging.log(f"loading model training data from {home_string}/{data_string}")

        data = pd.read_csv(f'{path}',index_col=0)


   
        X,y = self.feature_target_separator(data)

        return X,y
 
  


@dataclass
class Trainer(Utils):
    def __init__(self):
        super().__init__()
        self.sklearn_model = self.config["model"]["model_name"]
        self.mlflow_model = self.config["model"]["registered_model"]
        self.model_params = self.config["model"]["params"]
  

        pipeline = field(default_factory=self.create_pipeline())

 

    def create_pipeline(self):

        ### Function which return column transformers and end model for predictions ###

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
            )

        categorical_transformer = Pipeline(
            steps=[
        ("encoder", TargetEncoder())
            ]
            )
        
      

        preprocessor = ColumnTransformer(
            transformers=[
        ("numericalcols", numeric_transformer, self.config["num_features"]),
        ("categoricalcols", categorical_transformer, self.config["cat_features"])
        ] ,remainder='drop'
            )

        model_map = {
            'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
            'AdaBoostClassifier':AdaBoostClassifier
        }
    
        model_class = model_map[self.sklearn_model]
        model = model_class(**self.model_params)  

        model = AdaBoostClassifier(algorithm="SAMME")
        

        return preprocessor,model
    
    
    def save_model(self):

        logging.info("starting mlflow server from ")
        mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
        tracking_server_url = mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

        mlflow.set_experiment("rainfalL_prediction")
        mlflow_client = mlflow.tracking.MlflowClient(
        tracking_uri=tracking_server_url)

        preprocessor,model = self.create_pipeline()


        train_data = pd.read_csv(f'{train_path}',index_col=0)
                                 
        test_data = pd.read_csv(f'{test_path}',index_col=0)

        X_train = train_data.drop(["Date","day","Target_encoded","Location"],axis=1)
        y_train = train_data[self.target]

        logging.info(f"fitting pipeline to train data in {train_path}")
        
        preprocessor.fit(X_train,y_train)


        X_train_tr = preprocessor.transform(X_train)


        X_test = test_data.drop("Target_encoded",axis=1)
        y_test = test_data["Target_encoded"]
        X_test_tr = preprocessor.transform(X_test)
 
        f1_metric = make_scorer(f1_score)
        recall_metric = make_scorer(recall_score)




        model.fit(X_train_tr,y_train)

       

    

        metrics = { "recall_score": recall_metric,"f1_score":f1_metric}

        sfold = StratifiedKFold(n_splits=10,random_state=self.random_state,shuffle=True)
        logging.info("performing cross validation of model")
        cvs = cross_validate(model,X_test_tr,y_test,cv=sfold,scoring=metrics,n_jobs=-1)
        #insert stratified cross validate func and log the mean scores.
        
        scores =  {"recall_score":cvs["test_recall_score"].mean(),"f1_score":cvs["test_f1_score"].mean()}
        score_f1 = scores["f1_score"]
        logging.info(f"test f1 score was {score_f1}")
        print(score_f1)
        
        if score_f1 > 0.5:

            model.fit(X_train_tr,y_train)
            joblib.dump(model,f"{model_path}")
            joblib.dump(preprocessor,f"{preprocessor_path}")
       
            with mlflow.start_run():

                mlflow.log_metrics(scores)
                mlflow.log_params(model.get_params())

                signature = infer_signature(X_test_tr, model.predict(X_test_tr))
                model_uri = mlflow.get_artifact_uri(self.mlflow_model)

        # Logging model in mlflow           
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=X_train_tr,
                registered_model_name=self.mlflow_model,
                 )
            logging.info("model trained and logged in mlflow")
            
            latest_versions = mlflow_client.get_latest_versions(self.mlflow_model, stages=["None"])
            latest_model_version = latest_versions[0].version
            mlflow_client.transition_model_version_stage(self.mlflow_model,
                                                         latest_model_version,
                                                         "Staging",archive_existing_versions=True)

            logging.info(f"model {self.mlflow_model} with version {latest_model_version} transitioned to staging")
         
                
        else:
            logging.info(f"new model did not clear threshold with validation test f1 score of {score_f1}")
        
    
       








@dataclass
class MLflow(Trainer):

    def __init__(self):
        self.config = load_config()
        self.model_name = self.config["model"]["saved_model"]
        self.champion_name = self.config["model"]["registered_model"]

    def get_or_create_experiment(self,experiment_name:str):

        if experiment := mlflow.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        else:
   
            return mlflow.create_experiment(experiment_name)





    def tuning(self,objective,experiment_id,run_name):


        with mlflow.start_run(experiment_id=experiment_id,run_name=run_name,nested=True):
  # Initialize the Optuna study
          #  study = optuna.create_study(directions=['maximize', 'maximize', 'maximize', 'maximize'])
            study = optuna.create_study(directions=["maximize","maximize"])
  # Execute the hyperparameter optimization trials.
  # Note the addition of the `champion_callback` inclusion to control our logging
            study.optimize(objective, n_trials=50,n_jobs=-1)

            mlflow.log_params(study.best_trials.best_params)
            mlflow.log_metric("best f1 and maxwell coeff score", study.best_trials.best_value)

  # Log tags
            mlflow.set_tags(
        tags={
          "project": "australia weather forecasting",
          "optimizer_engine": "optuna",
          "model_family": "sklearn",
          "feature_set_version": 1,
            }
    	        )
        
        best_params = study.best_trials.params
        best_metrics = study.best_trials.values
        

        return best_params,best_metrics
    
    
def main():

    tr = Trainer()
    tr.save_model()
    print("new model registered and saved locally")
    logging.info("new model registered and saved locally")



if __name__ == "__main__":
    main()

