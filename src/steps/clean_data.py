import sys,os
sys.path.append('../') #allows us to import modules from dir one level above
import pandas as pd
#from category_encoders.woe import WOEEncoder
from sklearn.preprocessing import TargetEncoder
from core import config
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer




df_raw = pd.read_csv('../data/weatherAUS.csv',
                        skiprows=7,delimiter=",",
                        encoding = 'ISO-8859-1') 

categorical_features = df_raw.select_dtypes(include = ['object']).columns.tolist()
numerical_features = df_raw.select_dtypes(include = ['float64']).columns.tolist()


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
   
  target = TargetEncoder()
  

  encoded = target.fit_transform(X=df[categorical_features],y=df["RainTomorrow"])
  df_encoded = pd.DataFrame(encoded,categorical_features)

  #merge dataframs back

  df = pd.concat(df[numerical_features],df_encoded,axis=1)



def main():
    df_imputed = random_sample_imputation(df_raw)
    df_encoded = encode_cols()



