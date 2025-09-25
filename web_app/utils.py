
import yaml,os,joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics import make_scorer,recall_score,f1_score,matthews_corrcoef



train_file = 'model_training'
test_file = 'model_validation_data'
ref_file = 'monitoring_reference_data'

#home_dir = 'C:/Users/naras/OneDrive/Documents/weather_forecast_project/oct24_bmlops_int_weather'
#home_dir = '/web_app'
home_dir = '/opt/render/project'
raw_data_dir = 'src/input_data/training/weatherAUS.csv'
train_dir = 'src/output_data/train/model_training.csv'
test_dir = 'src/output_data/test/model_validation_data.csv'
ref_dir = 'src/output_data/ref/monitoring_reference_data.csv'
prod_dir = 'src/output_data/prod/monitoring_production_data.csv'
prod_predictions_dir = 'src/output_data/prod/prediction_production_data.csv'


reports_dir = 'reports'
model_dir = 'models/current_champion.pkl'
input_data_path = f'{home_dir}/src'
conf_dir = 'src/config.yml'
raw_data_path = f'{home_dir}/{raw_data_dir}'
train_path = f'{home_dir}/{train_dir}'
test_path = f'{home_dir}/{test_dir}'
ref_path = f'{home_dir}/{ref_dir}'
prod_path = f'{home_dir}/{prod_dir}'
prod_processed_path = f'{home_dir}/{prod_predictions_dir}'

conf_path = f'{home_dir}/{conf_dir}'
model_path = f'{home_dir}/{model_dir}'
preprocessor_path = f'{home_dir}/models/preprocessor.pkl'
location_encoder_path = f'{home_dir}/models/location_encoder.pkl'

reports_path = f'{home_dir}/{reports_dir}'

      



def load_config():
        
        with open(conf_path, 'r') as config_file:
            return yaml.safe_load(config_file.read())

