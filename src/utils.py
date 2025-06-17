
import yaml,os
from pathlib import Path
import pandas



train_file = 'model_training'
test_file = 'model_validation_data'
ref_file = 'monitoring_reference_data'


home_dir = 'C:/daten/numapathy/Dokumente/networksecurity_project/oct24_bmlops_int_weather'
raw_data_dir = 'src/input_data/training/weatherAUS.csv'
train_dir = 'src/output_data/train/model_training.csv'
test_dir = 'src/output_data/test/model_validation_data.csv'
ref_dir = 'src/output_data/ref/monitoring_reference_data.csv'
reports_dir = 'reports'
model_dir = 'models/current_champion.pkl'
input_data_path = f'{home_dir}/input_data'
conf_dir = 'src/config.yml'
raw_data_path = f'{home_dir}/{raw_data_dir}'
train_path = f'{home_dir}/{train_dir}'
test_path = f'{home_dir}/{test_dir}'
ref_path = f'{home_dir}/{ref_dir}'
conf_path = f'{home_dir}/{conf_dir}'
model_path = f'{home_dir}/{model_dir}'
reports_path = f'{home_dir}/{reports_dir}'





def load_config():
        
        with open(conf_path, 'r') as config_file:
            return yaml.safe_load(config_file.read())



