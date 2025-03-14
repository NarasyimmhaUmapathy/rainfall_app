import sys,os
sys.path.append('../')
import joblib
import yaml
import mlflow
from src.steps.ingest_data import load_data,clean

from sklearn.preprocessing import TargetEncoder, RobustScaler,OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import pandas as pd




class Trainer:
    def __init__(self):
        self.config = self.load_config()
        self.model_name = self.config["model"]["name"]
        self.model_params = self.config['model']['params']
        self.model_path = self.config['model']['store_path']
        self.categorical_features = self.config['cat_features']
        self.numerical_features = self.config['num_features']
        self.target = self.config['target']

        self.pipeline = self.create_pipeline()

    def load_config(self):
        with open('./config.yml', 'r') as config_file:
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
        X = data.drop("Target_encoded",axis=1)
        y = data["Target_encoded"]
        return X, y

    

    def train_model(self, X_train, y_train):
        pipeline = self.pipeline.fit(X_train, y_train)
        return pipeline

    

    def save_model(self):
        model_file_path = os.path.join(self.model_path,f'{self.model_name}.pkl')
        joblib.dump(self.pipeline, model_file_path)


trainer = Trainer()
pipe,processer = trainer.create_pipeline()
df,train,test = load_data()
X_train, y_train = trainer.feature_target_separator(train)
X_test,y_test = trainer.feature_target_separator(test)
pipe.fit(X_train,y_train)





