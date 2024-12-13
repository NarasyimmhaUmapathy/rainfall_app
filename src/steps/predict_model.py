# import functions to preprocess data and csv file
#import function to split data to x and y

# import grid searched params or best model

# predict incoming data with preprocessing pipeline and produce output

import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from ingest_data import load_data


class Predictor:
    def __init__(self):
        self.model_path = self.load_config()['model']['store_path']
        self.pipeline = self.load_model()

    def load_config(self):
        import yaml
        with open('../config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def load_model(self):
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        return joblib.load(model_file_path)

    def feature_target_separator(self, data):
        X = data.drop("RainTomorrow",axis=1)
        y = data["RainTomorrow"]
        return X, y

    def evaluate_model(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        #class_report = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        return accuracy, roc_auc
    

train,test = load_data()
predict = Predictor()
model = predict.load_model()
X,y = predict.feature_target_separator(test)
print(X.info())
scores = predict.evaluate_model(X,y)
print(scores)
