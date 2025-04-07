# 1. loads current production model from mlflow

# 2. monitors data drift and metric drift every month?

# 3. if data drift or metric drift on current month data passes certain tresholds, retrain model on current months data and perfom tuning again.
import sys
sys.path.append("../")

import mlflow,pandas as pd,joblib
from sklearn.metrics import matthews_corrcoef
import evidently
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset, DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently.report import Report
from datetime import datetime, date, time, timedelta
from evidently.test_preset import DataDriftTestPreset

from utils import train_path,test_path,ref_path,home_path


tracking_server_url=mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")



model = mlflow.sklearn.load_model("models:/test_monitoring/Staging")



test_data = pd.read_csv(f'{train_path}',index_col=0)
reference_data = pd.read_csv(f'{ref_path}',index_col=0)

def create_report(i: int):
    data_drift_report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            
        ]
                )

    data_drift_report.run(reference_data=reference_data, current_data=test_data)

# visualizing model decay

time_col = reference_data["year"]
true_values = reference_data["RainTomorrow"]
pred_values = model.predict(reference_data.drop("RainTomorrow",axis=1))
metric_matthews = matthews_corrcoef(true_values,pred_values)

values = {"time_col":time_col,
              "true_values":true_values,
              "pred_values":pred_values,
              "metric_matthews":metric_matthews}
decay_df = pd.DataFrame(values)


#visualize drifts in each feature and rank, compare with corresponding metric decay to check for correlation or relationship

def get_drift(test_data:pd.DataFrame,ref_data:pd.DataFrame) -> pd.DataFrame|None:
    
    #use different metrics for cat features with high cardinality
    #otherwise use standard metrics for num and cat features
    # rank top N features, return df of drift against time

    pass


print(decay_df.head())