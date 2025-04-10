# test pipeline output and metrics with sample data
import sys
sys.path.append('../')
from oct24_bmlops_int_weather.src.steps.train_model import Trainer
from oct24_bmlops_int_weather.src.utils import *


tr = Trainer()
model = tr.create_pipeline()

conf = load_config()
print(model.named_steps["model"])

ff = 5

assert ff == 4