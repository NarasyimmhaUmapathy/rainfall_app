# loads the train_model script, trains a new model pipeline and performs param tuning

# step 2 is logging the params and metrics and the best model in mlflow after it passes some tests

# this is the purpose of the main.py file, the streamlit app will be in the app.py file in a dev branch.
from steps.train_model import MLflow,Trainer
from steps.ingest_data import *
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import f1_score,matthews_corrcoef,roc_auc_score
from sklearn.ensemble import AdaBoostClassifier


from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

if __name__=="__main__":

    mlflow_obj = MLflow()
    trainer = Trainer()
    random_state = trainer.random_state
    tracking_server_url=mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    mlflow_client = mlflow.tracking.MlflowClient(
    tracking_uri=tracking_server_url)
    experiment_details = mlflow.get_experiment_by_name("australian_weather")
    id = experiment_details.experiment_id

    name = "test_monitoring"
   


    # test if model metrics pass certain targets, if yes log the model and compare with prod model, if yes then transition to production, if not none or staging


    def objective(trial):
        
        model = trainer.create_pipeline()
        X,y = trainer.load_data()
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=random_state,stratify=y)

        with mlflow.start_run(nested=True):

            params = {

        "learning_rate":trial.suggest_float("learning_rate",0.1,1),
        "max_depth" :trial.suggest_int("max_depth",2,10),
        "max_features":trial.suggest_float("max_features",0.5,1)
       

            }

    

        

            model.fit(X_train, y_train)

        #score = f1_score(y_test,model.predict(X_test),pos_label="Yes")
            maxwell_score:float = matthews_corrcoef(y_test,model.predict(X_test))
            precision_score:float = precision_score(y_test,model.predict(X_test))
            recall_score:float = recall_score(y_test,model.predict(X_test))
            f1:float = f1_score(y_test,model.predict(X_test),pos_label="Yes")


        #log each run to mlflow
        #mlflow.log_params(params)
        #mlflow.log_metric("f1_score",score)

      

        return maxwell_score,precision_score,recall_score,f1_score


    best_params,metrics = mlflow_obj.tuning(objective,id,"run with max features param")
    model_uri=mlflow_obj.save_model(best_params,"models",name)
    mlflow_client.transition_model_version_stage(name=name,version=1,stage="Staging")

    #run some test, if the model passes, promote to production




    





    





