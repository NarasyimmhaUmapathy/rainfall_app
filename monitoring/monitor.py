import pandas as pd,sys,joblib
import numpy as np
from sklearn.metrics import matthews_corrcoef,f1_score,precision_score,recall_score,roc_auc_score
from multiprocessing import Process,Pool
import asyncio
from datetime import datetime
sys.path.append("../")
from evidently.metric_preset import ClassificationPreset, DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric,DatasetCorrelationsMetric,ColumnCorrelationsMetric,ColumnDistributionMetric
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently.report import Report
from evidently import ColumnMapping
from src.utils import *

conf = load_config()
model = joblib.load(model_path)
test_data = pd.read_csv(f"{test_path}")
ref_data = pd.read_csv(f"{ref_path}")
#categorical_features = conf["cat_features"]
numerical_features = conf["num_features"]
ref_data["target"] = ref_data["Target_encoded"]
pred_values = pd.Series(model.predict(test_data.drop("Target_encoded",axis=1)))
pred_values_binary = pred_values.apply(lambda x: 1 if x == "Yes" else 0)

ref_data["prediction"] = pred_values_binary
ref_data_test = ref_data.query("Date >= '2017-01-01' \
                       and Date < '2017-03-26'")

# simulating production data as the last 3 months of reference data for the monitoring reports
prod_data = ref_data.query("Date >= '2017-03-26' \
                       and Date <= '2017-06-26'")


def return_mapping(target:str,prediction:str):

    column_mapping = ColumnMapping()

    column_mapping.target = target
    column_mapping.prediction = prediction
  #  column_mapping.prediction = ''
    column_mapping.numerical_features = numerical_features

    return column_mapping


def feature_metrics(batch_size:int,step_size:int):

    column_mapping = return_mapping("target","prediction")

   

    #column_mapping.categorical_features = categorical_features

    #column_mapping.datetime = "Date"
    data_drift_report = Report(metrics=[
                                    DataDriftPreset(num_stattest="jensenshannon",num_stattest_threshold=0.5,cat_stattest="chisquare",drift_share=0.5),
                                    ColumnDriftMetric(column_name="MinTemp",stattest_threshold=0.2),
                                    ColumnDriftMetric(column_name="MaxTemp",stattest_threshold=0.2),
                                    ColumnDriftMetric(column_name="Evaporation",stattest_threshold=0.2),
                                    ColumnDriftMetric(column_name="Sunshine",stattest_threshold=0.2),
                                   
                                    ]
)
    data_drift_report.run(reference_data=ref_data, current_data=prod_data[step_size * batch_size : (step_size + 1) * batch_size],column_mapping=column_mapping)
    report_path = '../reports/drift_report.html'
    data_drift_report.save_html(report_path)
    return report_path

def model_metrics(batch_size:int,step_size:int):

    date_time = datetime.today().strftime('%Y-%m-%d')

    column_mapping = return_mapping("target","prediction")


    #column_mapping.categorical_features = categorical_features

    #column_mapping.datetime = "Date"
    metric_drift_report = Report(metrics=[
                                    ClassificationPreset()
                                    
                                    ]
)
    metric_drift_report.run(reference_data=ref_data_test, current_data=prod_data[step_size * batch_size : (step_size + 1) * batch_size],column_mapping=column_mapping)
    report_path = f'../reports/metric_drift_report_{date_time}.html'
    current_f1_score = metric_drift_report.as_dict()["metrics"][0]["result"]["current"]["f1"]

    metric_drift_report.save_html(report_path)
    return report_path #to be used later in fastapi app in the endpoint
