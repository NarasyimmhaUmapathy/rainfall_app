import sys,os
sys.path.append('../') #allows us to import modules from dir one level aboe
import pandas as pd
from category_encoders.woe import WOEEncoder
from models.predict_model import random
from core import config
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest

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




#encode categorical features

def encode_cols(df,cols):
   
  woe = WOEEncoder()
  categorical_features = df.select_dtypes(include = ['object']).columns.tolist()
  numerical_features = df.select_dtypes(include = ['float64']).columns.tolist()

  df_encoded = woe.fit_transform()

  #merge dataframs back

  df_all = df.merge(df_encoded)





#scale features

def scaler(df):
   
   sc = RobustScaler()
   df_scaled = sc.fit_transform(df)
   return df_scaled,sc 





