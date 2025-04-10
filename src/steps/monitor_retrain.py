# 1. loads current production model from mlflow

# 2. monitors data drift and metric drift every month?

# 3. if data drift or metric drift on current month data passes certain tresholds, retrain model on current months data and perfom tuning again.
import sys
sys.path.append("../")

import mlflow,pandas as pd,joblib,os
import seaborn as sns
import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import matthews_corrcoef
import evidently,deepchecks
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
from evidently.test_suite import TestSuite
from evidently.ui.workspace import WorkspaceBase,Workspace

from evidently.ui.dashboards import CounterAgg
from evidently.ui.dashboards import DashboardPanelCounter
from evidently.ui.dashboards import DashboardPanelPlot
from evidently.ui.dashboards import PanelValue
from evidently.ui.dashboards import PlotType
from evidently.ui.dashboards import ReportFilter
from evidently.ui.dashboards import DashboardPanelTestSuite
from evidently.ui.dashboards import TestFilter
from evidently.ui.dashboards import TestSuitePanelType
from evidently.renderers.html_widgets import WidgetSize


from oct24_bmlops_int_weather.src.utils import train_path,test_path,ref_path,home_dir,load_config

conf = load_config()



WORKSPACE = "australian weather"
tracking_server_url=mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")



#model = mlflow.sklearn.load_model("models:/test_monitoring/Staging")
model_path = os.path.join(home_dir,'steps/model.pkl')
model = joblib.load(model_path)



test_data = pd.read_csv(f'{test_path}',index_col=0)
reference_data = pd.read_csv(f'{ref_path}',index_col=0)


batch_size = 200

def create_data_quality_report(i:int):
    data_drift_report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            DataDriftPreset(),
            ColumnDriftMetric(column_name="WindDir9am")
            
        ],timestamp=datetime.now() - timedelta(i),
        tags=[]
                )

    data_drift_report.run(reference_data=reference_data, current_data=test_data[i*batch_size:(i+1)*batch_size])
    return data_drift_report
def create_data_drift_test_suite(i: int):
    suite = TestSuite(
        tests=[
            DataDriftTestPreset()
        ],
        timestamp=datetime.now() + timedelta(days=i),
        tags = []
    )

    suite.run(reference_data=reference_data, current_data=test_data[i * batch_size : (i + 1) * batch_size])
    return suite


    
# have to predict every month, then save this value. not all of the reference data at once. then viz the decay with time with feature drift per month

year_col = reference_data["year"]
month_col = reference_data["month"]
true_values = reference_data["RainTomorrow"]
pred_values = model.predict(reference_data.drop("RainTomorrow",axis=1))


values = {"month":month_col,
          "year":year_col,
              "true_values":true_values,
              "pred_values":pred_values,
             # "metric_matthews":metric_matthews}
}
decay_df = pd.DataFrame(values)


#visualize drifts in each feature and rank, compare with corresponding metric decay to check for correlation or relationship

def get_drift(test_data:pd.DataFrame,ref_data:pd.DataFrame) -> pd.DataFrame|None:
    
    #use different metrics for cat features with high cardinality
    #otherwise use standard metrics for num and cat features
    # rank top N features, return df of drift against time

    pass

# get most important features
# plot model decay with feature drift and analyze what amount of drifts in each feature is contributing to the model deacay
# 



if __name__ == "__main__":
    print(matthews_corrcoef(true_values[:batch_size],pred_values[:batch_size]))
    print(matthews_corrcoef(true_values[:batch_size+batch_size],pred_values[:batch_size+batch_size]))

