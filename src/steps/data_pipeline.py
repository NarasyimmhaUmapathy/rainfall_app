
import sys
sys.path.append("../")
from ingest_data import *
from src.utils import *
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import ClassImbalance,FeatureLabelCorrelation,FeatureFeatureCorrelation
import great_expectations as ge     


input_train_df = load_raw_data(raw_data_path)
conf = load_config()
target = conf["target"]
target = ' '.join(target)


def check_path_exists(paths):

        for p in paths:
            if Path(f"{p}").exists():
                raise ValueError(f"{p} already exists")
            


    
def validate_data():

    
    # schema checks
  
    
    # correlation checks

    dataset = Dataset(input_train_df, label=target)
    feature_label_check = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9)
    neg_class,pos_class = input_train_df[target].value_counts(normalize=True)


    if  not feature_label_check.run(dataset=dataset):
        print("one or more features having correlation higher than threshold")
        return True


    elif pos_class/neg_class > 0.4:
        print("target class relationship has changed significantly")

        return True
    
def main():
    try:
        validate_data()
    except:
        print("data validation failed")

    else:
        paths = [train_path,test_path,ref_path]

      
    #logging splitting source data to model trianing, validation and monitoring reference datasets
        data_split(raw_data_path,paths)
    #logging data splitted to training, testing and monitoring sets

    
    
  

     
	    
    # Check validation results and raise an alert if validation fails


if "__main__" == __name__:
        main()