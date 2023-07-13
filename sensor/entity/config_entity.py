import os,sys
from sensor.exception import SensorException
from sensor.logger import logging
from datetime import datetime

FILE_NAME="sensor.csv"

TRAIN_FILE_NAME="train.csv"
TEST_FILE_NAME="test.csv"

class TrainingPipelineConfig:
    def __init__(self) -> None:
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception  as e:
            raise SensorException(e,sys)

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.database_name="aps"
        self.collection_name ="sensor"
        self.data_ingestion_dir =os.path.join(training_pipeline_config.artifact_dir,"data_ingestion")
        self.feature_store_file_path =os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
        self.train_file_path =os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
        self.test_file_path =os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
        self.test_size=.2
    def to_dict(self)->dict:
        try:
            return self.__dict__
        except Exception as e:
            raise SensorException(e,sys)




class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir =os.path.join(training_pipeline_config.artifact_dir,"data_validation")
        self.drop_null_value_thresold=.2
        self.base_file_path = os.path.join("aps_failure_training_set1.csv")
        self.report_file_path =os.path.join(self.data_validation_dir,"report.yaml")

class DataTransformationConfig:...
class ModelTrainerConfig:...
class ModelEvalutionConfig:...
class ModelPushConfig:...