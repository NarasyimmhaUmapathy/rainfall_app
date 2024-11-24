import sys,os
sys.path.append('../') #allows us to import modules from dir one level aboe
import pandas as pd
from models.predict_model import random

df = pd.read_csv('../data/weatherAUS.csv',
                        skiprows=7,delimiter=",",
                        encoding = 'ISO-8859-1') 

# handle missing data

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



# handle outliers

def outlier_handling(df,cols):




#encode categorical features

def encode_cols(df,cols):




#handle imbalanced target

def resample(df,target):



#scale features






#feature selection