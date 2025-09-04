
import sys
sys.path.append("../")
from ingest_data import *
from utils import *
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import ClassImbalance,FeatureLabelCorrelation,FeatureFeatureCorrelation
import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer





logger_ingestion = logging.getLogger("data_pipeline_logger")
logging.basicConfig(filename = f'{home_dir}/logs/data_pipeline.log',
                    level = logging.INFO,
                    format = '%(asctime)s:%(levelname)s:%(message)s')




conf = load_config()
target = conf["target"]
target = ' '.join(target)


def check_path_exists(paths):

        for p in paths:
            if Path(f"{p}").exists():
                raise ValueError(f"{p} already exists")
            


    
def validate_data(df):

  
    
    # correlation checks

    dataset = Dataset(df, label=target,cat_features=conf["cat_features_raw"])
    feature_label_check = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.8)
    result = feature_label_check.run(dataset)
    if not result:
        logging.warning("feature {a} is highly correlated with label")

        return True


            
    c_imb = ClassImbalance().add_condition_class_ratio_less_than(0.3)
    
    if not c_imb.run(dataset=dataset):
         logging.warning("target class is imbalanced")

         return True

    
def main():

    
    raw_data = load_raw_data(raw_data_path)
    if  validate_data(df=raw_data):
  
        print("data validation failed")
        logging.info("data validation failed")

    else: 
        paths = [train_path,test_path,ref_path]

      
    #logging splitting source data to model trianing, validation and monitoring reference datasets
    data_split(raw_data_path,paths)
    logging.info("data has been succesfully split,prod data is between '2017-01-01'and '2017-06-20'")

        
    #logging data splitted to training, testing and monitoring sets

    
    
  

     
	    
    # Check validation results and raise an alert if validation fails


if "__main__" == __name__:
    main()
    #paths = [train_path,test_path,ref_path]
    #data_split(raw_data_path,paths)

   # df_raw = load_raw_data(raw_data_path)
   


        