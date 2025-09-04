
import sqlite3 ,csv
import sys,os,yaml
sys.path.append('../../')

import pandas as pd,numpy as np
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, RobustScaler,OneHotEncoder,LabelEncoder
from steps.utils import *
from pathlib import Path
import logging,joblib
from enum import Enum
from pydantic import BaseModel,ValidationError



class WeatherData(BaseModel):
    Date : date
    MinTemp : float
    MaxTemp : float
    Rainfall : float
    Evaporation : float
    Sunshine : float
    WindGustSpeed : float
    WindSpeed9am : float
    WindSpeed3pm : float
    Humidity9am : float
    Humidity3pm : float
    Pressure9am : float
    Pressure3pm : float
    Cloud9am : float
    Cloud3pm : float
    Temp9am : float
    Temp3pm : float
    WindGustDir : str
    WindDir9am : str
    WindDir3pm : str
    RainToday : str
    Location : str


logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"{home_dir}/logs/data_ingestion.log", filemode="a",encoding='utf-8', level=logging.INFO,
                    
                    format='%(asctime)s %(name)s %(filename)s->%(funcName)s():%(lineno)s %(message)s ')





def load_config() -> yaml:
        with open(conf_path, 'r') as config_file:
            return yaml.safe_load(config_file.read())
        

def path_maker(dirs:list):

    config = load_config()
    home = config["directories"]["home"]
    home_string = " ".join(str(x) for x in home)


    train_string  = " ".join(str(x) for x in dirs[0])





   
    train_path = f"{home_string}/{train_string}"



    return train_path


    

def load_raw_data(path):

  
# Load CSV data into Pandas DataFrame 
    df = pd.read_csv(path,
                        delimiter=",",
                        encoding = 'ISO-8859-1') 
    
# check input columns with pydantic schema

    return df



def make_dirs():

    config = load_config()
    home = config["directories"]["home"]
    train_dir = config["directories"]["train"]
    test_dir = config["directories"]["test"]
    ref_dir = config["directories"]["ref"]

    all_dirs = [train_dir,test_dir,ref_dir]
    for a in all_dirs:
        a = " ".join(str(x) for x in a)

    


    data = [train_dir,test_dir,ref_dir]
    home_string = " ".join(str(x) for x in home)

    for d in data:
        aa = " ".join(str(x) for x in d)
        os.makedirs(os.path.join(home_string,aa))


def impute_encode(df):


    

    #use parquet, can scale for massive datasets or csv files


    cols_with_missing_values = df.columns[df.isna().any()].tolist()

    for var in cols_with_missing_values:

    # extract a random sample
        random_sample_df = df[var].dropna().sample(df[var].isnull().sum(),
                                                  random_state=0)
    # re-index the randomly extracted sample
        random_sample_df.index = df[
                df[var].isnull()].index

    # replace the NA
        df.loc[df[var].isnull(), var] = random_sample_df



    #encode location variable


    encoder_target = LabelEncoder()
    #woe = ce.WOEEncoder(cols=["Location"],random_state=144)
    encoder_location_column = TargetEncoder(target_type="binary")

    X = df["Location"].values.reshape(-1,1)
    y = df["RainTomorrow"]


    logging.info("encoding target and location columns")
    df_enc = encoder_location_column.fit_transform(X,y)
    encoded_target = encoder_target.fit_transform(y)

    array = np.array(df_enc)
    array_target = np.array(encoded_target)

    df["Location_encoded"] = pd.Series(array.flatten())
    df["Target_encoded"] = pd.Series(array_target.flatten())

    df.drop("RainTomorrow",axis=1,inplace=True)



    logging.info(f"saving location encoder in {home_dir}/models/location_encoder.pkl")
    joblib.dump(encoder_location_column,f"{home_dir}/models/location_encoder.pkl")


    #test output if everything have been encoded correctly and no NAs present



    return df



def data_split(input_train_data_path,output_paths):


    df_raw = pd.read_csv(input_train_data_path,
                        delimiter=",",
                        encoding = 'ISO-8859-1') 
    
    df = impute_encode(df_raw)




    
    #use GX to check data quality
        
    df["Date"] = pd.to_datetime(df["Date"])
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.weekday
    df["year"] = df["Date"].dt.year


    df.dropna(inplace=True)

    num_cols = df.columns.to_list()

    logging.info(f"dataframe with {num_cols} has been imputed and Location and RainTomorrow column has been encoded")
    

    #integrate test to prvent leakage, train not being test etc
    
    train_data = df.query("Date >= '2012-01-01' \
                       and Date < '2016-05-01'")
    test_data = df.query("Date >= '2016-05-01' \
                       and Date < '2016-12-31'")
    ref_data = df.query("Date >= '2017-01-01' \
                       and Date < '2017-06-20'")
    
    # simulating production data as the last 3 months of reference data
    prod_data = ref_data.query("Date >= '2017-03-26' \
                       and Date <= '2017-06-20'")
    
    
    

    # assert max date of train set is earlier than test set 
    if train_data["Date"].max() > test_data["Date"].min():
            raise AssertionError("max train date is greater than min test date!")
    if test_data["Date"].max() > ref_data["Date"].min():
            raise AssertionError("max test date is greater than min reference date!")


    conf = load_config()
    cols = conf["num_features"] + conf["cat_features"] 

   # if len(cols) != 21:
    #     raise AssertionError("number of input columns has changed from 21")
    
    logging.info(f"loading preprocessor to transform production data for api endpoint from {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    #X_train = train_data.drop(["Date","day","Target_encoded","Location","month"],axis=1)
    X_train = train_data[cols]
    y_train = train_data[conf["target_encoded"]]
    preprocessor.fit(X_train,y_train)
    logging.info("pre processing production data for model predictions with column transformer")
    prod_data_tr = preprocessor.transform(prod_data)
    prod_data_preprocessed = pd.DataFrame(prod_data_tr,columns=cols)
    #prod_data_preprocessed["Target_encoded"] = prod_data["Target_encoded"]
    prod_data_preprocessed["Location_encoded"] = prod_data["Location_encoded"]
    prod_data_preprocessed["Location"] = prod_data["Location"]
    prod_data_preprocessed["Date"] = prod_data["Date"]

    





    train_data.to_csv(f'{train_path}')
    test_data.to_csv(f'{test_path}')
    ref_data.to_csv(f'{ref_path}')
    prod_data.to_csv(f'{prod_path}')
    prod_data_preprocessed.to_csv(f'{prod_processed_path}')


    #logging that data has been split
    logging.info("data has been split to train,test,ref and prod dirs")

 






   