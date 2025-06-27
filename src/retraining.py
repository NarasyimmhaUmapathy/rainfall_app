# loads the train_model script, trains a new model pipeline and performs param tuning

# step 2 is logging the params and metrics and the best model in mlflow after it passes some tests

# this is the purpose of the main.py file, the streamlit app will be in the app.py file in a dev branch.
from steps.train_model import MLflow,Trainer
from steps.ingest_data import *
from steps.predict_model import Model
from monitor import feature_metrics
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import f1_score,matthews_corrcoef,roc_auc_score,make_scorer,average_precision_score,recall_score
from sklearn.ensemble import AdaBoostClassifier
import logging
from datetime import datetime
import joblib
from functools import lru_cache

from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{home_dir}/logs/retraining_pipeline.log', encoding='utf-8', level=logging.INFO,format='%(asctime)s %(message)s')


matthews_scores = make_scorer(matthews_corrcoef)
f1_scores = make_scorer(f1_score)
avg_prec_metric = make_scorer(average_precision_score)
recall_metric = make_scorer(recall_score)


evaluator = Model()
mlflow_object = MLflow()
tracking_server_url=mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow_client = mlflow.tracking.MlflowClient(
tracking_uri=tracking_server_url)
name = "test_monitoring"
#experiment_details = mlflow.get_experiment_by_name("australian_weather")
id = mlflow_object.get_or_create_experiment("rainfall_prediction")



def retrain_model(data:pd.DataFrame,drift_share:float):


    date_time = datetime.today().strftime('%Y-%m-%d')
    model_name = evaluator.registered_model
    logging.info(f"loading model with name: {model_name} and version: Staging from mlflow registry")
    try:
        model_current = mlflow.sklearn.load_model(f"models:/{model_name}/Staging")
    except ImportError as e:
        logging.warning(f"model with name {model_name} could not be loaded")
        
    #model_current = joblib.load(f"{model_path}")



    with mlflow.start_run(experiment_id=id, run_name=f'model retraining with share of drifted features {drift_share}, runtime:{date_time}'):           

        f1_score,recall_score = evaluator.evaluate_model(model_current,data)
        print(f"evaluated f1 score is {f1_score}")
        logging.info(f"test validation score is {f1_score}]")

        metrics = {"f1":f1_score,"recall":recall_score}
        mlflow.log_metrics(metrics)

        if f1_score > 0.5:               

            evaluator.update_model(model_current,prod_path)
            latest_versions = mlflow_client.get_latest_versions(model_name, stages=["None"])
            latest_model_version = latest_versions[0].version
            mlflow_client.transition_model_version_stage(evaluator.registered_model,
                                                         latest_model_version,
                                                         "Staging",archive_existing_versions=True)
            logging.info(f"model {evaluator.registered_model} with version {latest_model_version} transitioned to staging")
            return True     
        
        else:
            logging.info(f"retrained model did not clear threshold score with evaluated f1 score of {f1_score}")

            return False
    
def objective(trial):
        
        #model = trainer.create_pipeline()
        model = evaluator.current_model
        X,y = evaluator.load_data(f"{train_path}")
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=evaluator.random_state,stratify=y)

        with mlflow.start_run(nested=True):

            params = {

        "learning_rate":trial.suggest_float("learning_rate",0.1,1),
        "max_depth" :trial.suggest_int("max_depth",2,10),
        "max_features":trial.suggest_float("max_features",0.5,1)
       

            }

    

        

            model.fit(X_train, y_train)

            f1_scores = f1_score(y_test,model.predict(X_test),pos_label=1)
            matthews_scores:float = matthews_corrcoef(y_test,model.predict(X_test))
           # precision_score:float = precision_score(y_test,model.predict(X_test))
           # recall_score:float = recall_score(y_test,model.predict(X_test))
             #f1:float = f1_score(y_test,model.predict(X_test),pos_label="Yes")

        params = {"score_f1":f1_scores,"matthews_score":matthews_scores}
        #log each run to mlflow
        #mlflow.log_params(params)
        #mlflow.log_metric("f1_score",score)
      

      #  return maxwell_score,precision_score,recall_score,f1
        return f1_scores,matthews_scores

def parameter_tuning(run_name):

    
    current_model = evaluator.current_model
    random_state = evaluator.random_state
    

    mlflow_client = mlflow.tracking.MlflowClient(
    tracking_uri=tracking_server_url)
    experiment_details = mlflow.get_experiment_by_name("rainfall_prediction")
    id = experiment_details.experiment_id

   


    # test if model metrics pass certain targets, if yes log the model and compare with prod model, if yes then transition to production, if not none or staging

   


    best_params,metrics = mlflow_object.tuning(objective,id,run_name)
    #best_trials = mlflow_object.tuning(objective,id,run_name)


    mlflow.log_params(best_params)
    mlflow.log_metrics(metrics)

   # return best_params,metrics
    return best_params,metrics


    #model_uri=mlflow_obj.save_model(best_params,"models",name)
   # mlflow_client.transition_model_version_stage(name=name,version=1,stage="Staging")

    #run some test, if the model passes, promote to production

@lru_cache
def monitoring_data_preparation(interval_range:int):

    ### prepares reference and returns sliced bimonthly current data for evidently reports ###
    
    current_data = pd.read_csv(f"{prod_path}")
    ref_data = pd.read_csv(f"{ref_path}")
    model = joblib.load(f"{model_path}")
    preprocessor = joblib.load(f"{preprocessor_path}")

    pred_values = pd.Series(model.predict(preprocessor.transform(current_data)))

    #pred_values_binary = pred_values.apply(lambda x: 1 if x == "Yes" else 0)

    
    current_data["prediction"] = pred_values
    current_data["target"] = current_data["Target_encoded"]


    return ref_data,current_data
    ### returns reference and current data ###



### main retraining function on feature drift ###
def main(interval_range:int):

    ref_data,current_data = monitoring_data_preparation(interval_range)
    model = joblib.load(f"{model_path}")

    

    report = feature_metrics(interval_range,current_data)[1]
    
    dataset_drift = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
    drift_share = report.as_dict()["metrics"][0]["result"]["share_of_drifted_columns"]

    print(f"drift share is {drift_share},drift status is {dataset_drift}")

    if dataset_drift:
        logging.warning(f"dataset drift detected,{drift_share*100}% of total columns have drifted")
        print("dataset drift detected")



        if retrain_model(current_data,drift_share=drift_share):
            logging.info("Model retraining completed")
            print("model retraining completed")

        else:
            logging.warning("Model retraining failed!")
            print("model retraining failed")
    else:
        print(f"drift share of features was {drift_share},no retraining needed")
    


   
if "__main__" == __name__:

    main(1)