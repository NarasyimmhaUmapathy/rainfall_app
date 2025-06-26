import gradio as gr
import joblib

from steps.utils import *
from monitor import feature_metrics,prod_data

model = joblib.load(f"{model_path}")
conf = load_config()

model_features = conf["cat_features"] + conf["num_features"]
report_features = model_features + conf["target_encoded"]

def predict_rain(num_days:int):

    prediction = model.predict_proba(prod_data[30 * num_days: 30 * (num_days + 1)])

    if prediction > 0.5:
        return f"rainfall prediction for tommorow is high!"
    
    else:
        return f"rainfall prediction for tommorow is low!"
    

def feature_drift(num_days:int):
    return feature_metrics(num_days)





iface = gr.Interface(
   fn=feature_drift,
   inputs=[
       gr.Slider(
           minimum=0,
           maximum=3,
           step=1
           
       )
   ],
   outputs="text",
   title="Rain prediction for tommorow",
   description="figure out the rain chances for tommorow",
)
iface.launch(share=True)

