# loads the train_model script, trains a new model pipeline and performs param tuning

# step 2 is logging the params and metrics and the best model in mlflow after it passes some tests

# this is the purpose of the main.py file, the streamlit app will be in the app.py file in a dev branch.
from steps.train_model import MLflow,Trainer
from steps.ingest_data import *
from steps.predict_model import Model
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import f1_score,matthews_corrcoef,roc_auc_score,make_scorer,average_precision_score,recall_score
from sklearn.ensemble import AdaBoostClassifier
import logging
from datetime import datetime


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
id = mlflow_object.get_or_create_experiment("australian_weather")



def retrain_model(current_model_name,new_model_name,cv_splits:int):

    
    logging.info(f"loading model with name: {current_model_name} and version: Staging from mlflow registry")
    model_current = mlflow.sklearn.load_model(f"models:/{current_model_name}/Staging")

    
    logging.info(f"evaluating model with cross validation")

    eval_scores = evaluator.evaluate_model(model_current,test_path,cv_splits)
    if eval_scores[2] > 0.5: #load from config
        evaluator.update_model(model_current,train_path,new_model_name)
        return True
        
    else:
        logging.info("retrained model did not clear threshold score")
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
    experiment_details = mlflow.get_experiment_by_name("australian_weather")
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




def main(updated_model_name):
 pass


    #retrain model, validate and updte if validation passes
  #  if retrain_model("test_monitoring",updated_model_name,5):
    # tune and update model params with best params
   #     logging.info("performing hyperparameter tuning")
    #    best_params,metrics = parameter_tuning()
     #   evaluator.update_params(best_params,"models",updated_model_name)
      #  logging.info("updated model parameters in mlflow")

   
if "__main__" == __name__:
    date_time = datetime.today().strftime('%Y-%m-%d')
    model_uri = f"models:/{evaluator.registered_model}/1"
    #model = mlflow.sklearn.load_model(model_uri)
    mlflow_client.transition_model_version_stage(evaluator.registered_model,2,
                                                         "Staging",archive_existing_versions=True)
         
    sys.exit()
    
    model_current = mlflow.sklearn.load_model(f"models:/{evaluator.registered_model}/Staging")
    scores = evaluator.evaluate_model(model_current,test_path,5)
    if scores["test_f1_score"].mean() > 0.5:


        with mlflow.start_run(experiment_id=id, run_name=f'retraining_{date_time}'):

            logging.info("loading registered model from mlflow in main.py")


      #mlflow.log_metric("mean cv f1 score",scores["test_f1_score"].mean())
      #mlflow.log_metric("mean cv recall score",scores["test_recall_score"].mean())
      #mlflow.log_metric("mean cv precision score",scores["test_precision_score"].mean())

               
            logging.info("updating registered model with new production data and registering new version in mlflow")
            #evaluator.update_model(model_current,train_dir,test_dir)
           


      #log feature importances

      




    





