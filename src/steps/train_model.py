import sys,os
import joblib
import yaml

from sklearn.preprocessing import TargetEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeClassifier
from imblearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import pandas as pd
from ingest_data import load_data





class Trainer:
    def __init__(self):
        self.config = self.load_config()
        self.model_name = self.config["model"]["name"]
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
            transformers=[  
       # ("impute_categorical",SimpleImputer(strategy="most_frequent"),self.categorical_features),
        ('scale',RobustScaler(),self.numerical_features),
        ("impute_numerical",SimpleImputer(),self.numerical_features)
            ],remainder="passthrough"
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
        X = data.drop("RainTomorrow",axis=1)
        y = data["RainTomorrow"]
        return X, y

    def train_model(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def save_model(self):
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        joblib.dump(self.pipeline, model_file_path)




trainer = Trainer()
train,test = load_data()
X,y = trainer.feature_target_separator(train)
print(test.info())
#trainer.train_model(X,y)
#trainer.save_model()