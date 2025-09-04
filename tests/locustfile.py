from locust import FastHttpUser, task
import sys
sys.path.append('../')
from src.steps.utils import *

prod_data = pd.read_csv(f"{prod_path}",index_col=0)


class HelloWorldUser(FastHttpUser):
    #@task
    #def predict_rain(self):

     #   for i in prod_data["Location"].unique():
      #      self.client.get(f"/predict/{i}")

    @task()
    def predict_dashboard(self):
       self.client.get(f"/australia_rainfall_chance")



