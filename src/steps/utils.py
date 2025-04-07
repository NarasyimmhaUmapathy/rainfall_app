
import yaml,os
from pathlib import Path
import pandas

home_dir = 'C:/daten/numapathy/Dokumente/networksecurity_project/oct24_bmlops_int_weather/src'
train_dir = 'data/train/train_data.csv'
test_dir = 'data/test/test_data.csv'
ref_dir = 'data/ref/ref_data.csv'
train_path = f'{home_dir}/{train_dir}'
test_path = f'{home_dir}/{test_dir}'
ref_path = f'{home_dir}/{ref_dir}'
home_path = f'{home_dir}/{train_dir}'




def load_config():
        with open(home_path/'config.yml', 'r') as config_file:
            return yaml.safe_load(config_file.read())

def return_paths() -> list[Path]:
     
     pass


