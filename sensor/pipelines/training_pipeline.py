from sensor.logger import logging
from sensor.exception import SensorException
from sensor.utils import get_collection_as_dataframe
import sys,os
from sensor.entity import config_entity
from sensor.components.data_ingestion import DataIngestion
from sensor.components.Data_validation import DataValidation
from sensor.components.Model_trainer import ModelTrainer
from sensor.components.model_evaluater import ModelEvaluater
from sensor.components.model_pusher import ModelPusher
from sensor.components.data_transformation import DataTransformation


def start_training_pipeline():
    try:
        training_pipeline_config=config_entity.TrainingPipelineConfig()

        # data ingestion
        data_ingestion_config=config_entity.DataIngestionConfig(training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion=DataIngestion(data_ingestion_config)
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()

        # data validation
        data_validation_config=config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation =DataValidation(data_validation_config=data_validation_config,data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact=data_validation.initiate_data_validation()

        # data transformation
        data_transformation_config =config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)

        data_transformation=DataTransformation(data_transformation_config=data_transformation_config,data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact=data_transformation.initiate_data_transformation()

        # model trainer
        model_trainer_config =config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)

        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact= model_trainer.initiate_model_trainer()

        # model evaluation
        model_eval_config =config_entity.ModelEvalutionConfig(training_pipeline_config=training_pipeline_config)
        model_evaluater =ModelEvaluater(model_evalution_config=model_eval_config,data_ingestion_artifact=data_ingestion_artifact,
                                        data_transformation_artifact=data_transformation_artifact,model_trainer_artifact=model_trainer_artifact)
        
        model_evaluater_artifact =model_evaluater.initiate_model_evalution()
        
        # model pusher
        model_pusher_config =config_entity.ModelPushConfig(training_pipeline_config=training_pipeline_config)
        
        model_pusher =ModelPusher(model_pusher_config=model_pusher_config,data_transformation_artifact=data_transformation_artifact,model_trainer_artifact=model_trainer_artifact)
        model_pusher_artifact = model_pusher.initiate_model_push()
    except Exception as e:
        raise SensorException(e,sys)
