
import sqlite3 ,csv
import sys,os,yaml
sys.path.append('../')

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_config() -> yaml:
        with open('./config.yml', 'r') as config_file:
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

def load_data():

    df = pd.read_csv('./data/weatherAUS.csv',
                        delimiter=",",
                        encoding = 'ISO-8859-1') 
    

    
    df["Date"] = pd.to_datetime(df["Date"])
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.weekday

    config = load_config()

    df.dropna(inplace=True)
    df.drop("Date",axis=1,inplace=True)
    

    
    df_train = df.loc[:80000]
    df_test = df.loc[80001:]

    return df_train,df_test

def clean(df):

    trainer = Trainer()

    config = trainer.load_config()


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

