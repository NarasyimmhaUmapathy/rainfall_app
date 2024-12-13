import sys,os

import logging
import yaml
#import mlflow
#import mlflow.sklearn
from steps.ingest_data import load_data
from steps.train_model import Trainer
from steps.predict_model import Predictor
from sklearn.metrics import classification_report
from steps.train_model import Trainer
from steps.predict_model import Predictor

# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    # Load data
   
    train, test = load_data()
    logging.info("Data ingestion completed successfully")

    # Clean data

    #logging.info("Data cleaning completed successfully")

    # Prepare and train model
    trainer = Trainer()
    X_train, y_train = trainer.feature_target_separator(train)
    trainer.train_model(X_train, y_train)
    trainer.save_model()
    logging.info("Model training completed successfully")

    # Evaluate model
    predictor = Predictor()
    X_test, y_test = predictor.feature_target_separator(test)
    accuracy, class_report, roc_auc_score = predictor.evaluate_model(X_test, y_test)
    logging.info("Model evaluation completed successfully")
    
    # Print evaluation results
    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {trainer.model_name}")
    print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
    print(f"\n{class_report}")
    print("=====================================================\n")


        
if __name__ == "__main__":
    train, test = load_data()
    trainer = Trainer()
    X_train, y_train = trainer.feature_target_separator(train)
    trainer.train_model(X_train, y_train)




     
