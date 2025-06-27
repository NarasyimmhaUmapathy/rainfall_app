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
from steps.utils import *
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

conf = load_config()
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)


test_data = pd.read_csv(f"{test_path}")
ref_data = pd.read_csv(f"{ref_path}")
categorical_features = conf["cat_features"]
numerical_features = conf["num_features"]
ref_data["target"] = ref_data["Target_encoded"]

#pred_values = pd.Series(model.predict(test_data.drop("Target_encoded",axis=1)))
pred_values = pd.Series(model.predict(preprocessor.transform(ref_data)))

pred_values_binary = pred_values.apply(lambda x: 1 if x == "Yes" else 0)

ref_data["prediction"] = pred_values_binary

# simulating production data as the last 3 months of reference data for the monitoring reports
prod_data = ref_data.query("Date >= '2017-03-26' \
                       and Date <= '2017-06-25'")




def return_mapping(target:str,prediction:str):

    column_mapping = ColumnMapping()

    column_mapping.target = target
    column_mapping.prediction = prediction
  #  column_mapping.prediction = ''
    column_mapping.numerical_features = numerical_features


    return column_mapping


def feature_metrics(num_days:int,data):

    column_mapping = return_mapping("target","prediction")

   

    #column_mapping.categorical_features = categorical_features

    #column_mapping.datetime = "Date"
    data_drift_report = Report(metrics=[
                                    DataDriftPreset(num_stattest="jensenshannon",num_stattest_threshold=0.5,cat_stattest="chisquare",drift_share=0.01),
                                    ColumnDriftMetric(column_name="target",stattest_threshold=0.2),
                                    
                                    ]
)
    #data_drift_report.run(reference_data=ref_data, current_data=prod_data[step_size * batch_size : (step_size + 1) * batch_size],column_mapping=column_mapping)
    data_drift_report.run(reference_data=ref_data, current_data=data.iloc[14 * num_days: 14 * (num_days + 1), :],column_mapping=column_mapping)
    report_path = f'{home_dir}/reports/drift_report.html'
    data_drift_report.save_html(report_path)
    return report_path,data_drift_report

