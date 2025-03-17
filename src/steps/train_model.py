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
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score,matthews_corrcoef,roc_auc_score
import pandas as pd
import category_encoders as ce
import optuna


class MLflow:

    def __init__(self,experiment_name):
        self.config = self.load_config()
        self.experiment_name = experiment_name
        self.experiment_id = self.get_or_create_experiment()
        self.model = self.initial_model()
        self.model_name = self.config["model"]["name"]
        self.local_model_path = self.config["model"]["store_path"]

    def load_initial_model(self) -> joblib:

        model_file_path = os.path.join(self.local_model_path,f'{self.model_name}.pkl')
        model = joblib.load(model_file_path)
        return model


    def objective(self,trial):
        with mlflow.start_run(nested=True):
      # Define hyperparameters
            params = {
       
          "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
          "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
          "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            }
        
        
            trainer = Trainer()
            data = pd.read_csv(f'{trainer.home_string}/{self.config["data"]["train_path"]}')
            

            model = trainer.create_pipeline()
            learning_rate = trial.suggest_uniform("lr",0.1,1)
            max_iter = trial.suggest_uniform("max_iter",50,250)
            max_depth = trial.suggest_uniform("max_depth",5,15)

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            score = f1_score(y_pred,y_test)


    def get_or_create_experiment(self,experiment_name):

        if experiment := mlflow.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name)
        
    




class Trainer:
    def __init__(self):
        self.config = self.load_config()
        self.home_dir = self.config["directories"]["home"]
        self.model_name = self.config["model"]["name"]
        self.model_params = self.config['model']['params']
        self.model_path = self.config['model']['store_path']
        self.categorical_features = self.config['cat_features']
        self.numerical_features = self.config['num_features']
        self.target = self.config['target']
        self.test_ratio = self.config["train_test_ratio"]
        self.random_state = self.config["random_state"]

        self.pipeline = self.create_pipeline()
        

    def load_config(self):
        with open('../config.yml', 'r') as config_file:
            return yaml.safe_load(config_file.read())
        
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
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier
        }
    
        model_class = model_map[self.model_name]
        model = model_class(**self.model_params)


        pipeline = Pipeline([
        ('processor',processor),
        ('model',model)
        ])
        
 

        return pipeline

    def feature_target_separator(self, data):
        X = data.drop(self.config["target"],axis=1)
        y = data[self.config["target"]]
        return X, y
    
    
 

    

    def param_tuning(self,train_file:str):

        mlflow.set_tracking_uri("http://localhost:8080")

        experiment_name = "model_search_australian_weather"

        mlflow_obj = MLflow()

            
        experiment_id = mlflow_obj.get_or_create_experiment(experiment_name)

        mlflow.set_experiment(experiment_id=experiment_id)


        # compare models with mlflow and log the metrics, then perform param tuning with optuna on final model
        home_dir = self.home_dir
        train_data = pd.read_csv(f'{home_dir}/data/train/{train_file}.csv')
        X,y = self.feature_target_separator(train_data)
        
       

    def save_model(self):
        model_file_path = os.path.join(self.model_path,f'{self.model_name}.pkl')
        joblib.dump(self.pipeline, model_file_path)



trainer = Trainer()
model = trainer.create_pipeline()
home_string  = " ".join(str(x) for x in trainer.home_dir)


data = pd.read_csv(f'{home_string}data/train/train_data.csv',index_col=0)
data_val = pd.read_csv(f'{home_string}data/test/test_data.csv',index_col=0)


X,y = trainer.feature_target_separator(data)

def objective(trial):
    learning_rate = trial.suggest_uniform("learning_rate",0.1,1)
    max_depth = trial.suggest_int("max_depth",2,10)
    max_features = trial.suggest_int("max_features",1,5)
    max_bins = trial.suggest_int("max_bins",100,350)
    
    model = trainer.create_pipeline()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=trainer.random_state,stratify=y)

    model.fit(X_train, y_train)

    score = f1_score(y_test,model.predict(X_test),pos_label="Yes")
    return score
##trainer.save_model()
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)



