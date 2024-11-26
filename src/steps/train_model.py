import sys,os

import joblib
import yaml

from sklearn.preprocessing import TargetEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeClassifier
from imblearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

rf = RandomForestClassifier()
dtc = DecisionTreeClassifier()
ridge = RidgeClassifier()

models = {"random_forest":rf,"decision_classifier":dtc,"ridge_classifier":ridge}


class Trainer:
    def __init__(self):
        self.config = self.load_config()
        self.models = models
        self.model_params = self.config['model']['params']
        self.model_path = self.config['model']['store_path']
        self.categorical_features = self.config['cat_features']
        self.numerical_features = self.config['num_features']
        self.pipeline = self.create_pipeline()

    def load_config(self):
        with open('../config.yml', 'r') as config_file:
            return yaml.safe_load(config_file.read())
        
    def create_pipeline(self):
        processor = ColumnTransformer(       
        
        ('scale',RobustScaler())
        )

        pipeline = Pipeline([
        ('processor',processor,),
        ('model',self.models["random_forest"])
        ])
        
        
        model_map = {
            'RandomForestClassifier': RandomForestClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier
        }
    
 

        pipeline = Pipeline([
            ('preprocessor', processor),
            ('model', self.models["random_forest"])
        ])

        return pipeline

    def feature_target_separator(self, data):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y

    def train_model(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def save_model(self):
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        joblib.dump(self.pipeline, model_file_path)


tr = Trainer()
mod = tr.models
print(mod["ridge"])