import os,sys
import pandas as pd
from sensor.config import mongo_client
from sensor.logger import logging
from sensor.exception import SensorException
import yaml

def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
   
    return:
    Pandas dataframe of a collection
    """
    try:
        logging.info(f"reading data from mongodb database name : {database_name} and collection : {collection_name}")
        df =pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"faund colums: {df.columns}")
        if "_id" in df.columns:
            logging.info("droping column: _id")
            df = df.drop("_id",axis=1)
        logging.info(f"Row and columns in df: {df.shape}")
        return df
    except Exception as e:
        raise SensorException(e,sys)
def write_yml_file(file_path,data:dict):
        try:
            file_dir=os.path.dirname(file_path)
            os.makedirs(file_dir,exist_ok=True)
            with open(file_path,"w") as file:
                yaml.dump(data,file)
                
        except Exception as e:
            raise SensorException(e,sys)
def convert_column_to_float(df:pd.DataFrame,exclude_columns:list):
     try:
        for column in df.columns:
            if column not in exclude_columns:
                df[column]=df[column].astype('float')
        return df
     except Exception as e:
        raise SensorException(e,sys)