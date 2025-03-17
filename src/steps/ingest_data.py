
import sqlite3 ,csv
import sys,os,yaml
sys.path.append('../')

import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, RobustScaler,OneHotEncoder,LabelEncoder
import category_encoders as ce



from pathlib import Path


def load_config() -> yaml:
        with open('../config.yml', 'r') as config_file:
            return yaml.safe_load(config_file.read())

def ingest():
    with sqlite3.connect("./data/rainfall.db") as conn:
    # interact with database
        print(f"Opened SQLite database with version {sqlite3.sqlite_version} successfully.")

  
# Load CSV data into Pandas DataFrame 
    stud_data = pd.read_csv('./data/weatherAUS.csv',
                        delimiter=",",
                        encoding = 'ISO-8859-1') 
    
    test_data = pd.read_csv('./data/weatherAUS.csv',
                        delimiter=",",
                        encoding = 'ISO-8859-1')

    
# Write the data to a sqlite table 
    stud_data.to_sql('rainfall', conn, if_exists='replace', index=False) 
  
# Create a cursor object 
    cur = conn.cursor() 
# Fetch and display result 
    for row in cur.execute('SELECT * FROM rainfall LIMIT 20 '): 
        print(row) 
# Close connection to SQLite database 
    conn.close() 

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




def data_split():

    df = pd.read_csv('../data/weatherAUS.csv',
                        delimiter=",",
                        encoding = 'ISO-8859-1') 
        
    df["Date"] = pd.to_datetime(df["Date"])
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.weekday
    df["year"] = df["Date"].dt.year

    config = load_config()

    df.dropna(inplace=True)
    df.drop("Date",axis=1,inplace=True)
    


    
    df_train = df.loc[:40000]
    df_test = df.loc[40001:80000]
    df_ref = df.loc[80001:]

    train_file = 'train_data'
    test_file = 'test_data'
    ref_file = 'ref_data'
    config = load_config()
    home = config["directories"]["home"]
    home_string = " ".join(str(x) for x in home)
 

    

    df_train.to_csv(f'{home_string}/data/train/{train_file}.csv')
    df_test.to_csv(f'{home_string}/data/test/{test_file}.csv')
    df_ref.to_csv(f'{home_string}/data/ref/{ref_file}.csv')

 


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


    target = TargetEncoder(target_type="binary")
    #woe = ce.WOEEncoder(cols=["Location"],random_state=144)
    #label = LabelEncoder()

    X = df["Location"].values.reshape(-1,1)
    y = df["RainTomorrow"]

    df_enc = target.fit_transform(X,y.ravel())
    #encoded_target = label.fit_transform(y)

    array = np.array(df_enc)
    #array_target = np.array(encoded_target)

    df["Location_encoded"] = pd.Series(array.flatten())
   #df["Target_encoded"] = pd.Series(array_target.flatten())

    df.drop(["Location","RainTomorrow"],axis=1,inplace=True)


    return df



def random_sample_imputation(df):

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

    return df

