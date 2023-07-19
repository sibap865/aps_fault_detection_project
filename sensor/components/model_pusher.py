from sensor.predicter import ModelResolver
from sensor.exception import SensorException
from sensor.logger import logging
import os,sys
from sensor.utils import load_object,save_object

from sensor.entity.config_entity import ModelPushConfig
from sensor.entity.artifacts_entity import ModelPushArtifact,DataTransformationArtifact,ModelTrainerArtifact

class ModelPusher:

        def __init__(self,model_pusher_config:ModelPushConfig,
                     data_transformation_artifact:DataTransformationArtifact,
                     model_trainer_artifact:ModelTrainerArtifact) -> None:
                try:
                    logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
                    self.model_pusher_config=model_pusher_config
                    self.data_transformation_artifact=data_transformation_artifact
                    self.model_trainer_artifact =model_trainer_artifact
                    self.model_resolver=ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
                except Exception as e:
                    raise SensorException(e,sys)
        def initiate_model_push(self)->ModelPushArtifact:
                try:
                    # load_object
                    logging.info("loading model, transrormer and target encoder")

                    transformer =load_object(file_path=self.data_transformation_artifact.transformation_oject_path)
                    model=load_object(file_path=self.model_trainer_artifact.model_path)
                    target_encoder =load_object(file_path=self.data_transformation_artifact.target_encoder_path)

                    # model_pusher_dir
                    logging.info("saving model to model pusher dir")
                    save_object(file_path=self.model_pusher_config.pusher_transformer_path,obj=transformer)
                    save_object(file_path=self.model_pusher_config.pusher_model_path,obj=model)
                    save_object(file_path=self.model_pusher_config.pusher_target_encoder_path,obj=target_encoder)


                    # saved_model_dir
                    logging.info("saving model in saved model dir")
                    transformer_path =self.model_resolver.get_latest_save_transformer_path()
                    model_path =self.model_resolver.get_latest_save_model_path()
                    target_encoder_path=self.model_resolver.get_latest_save_target_encoder_path()
                    save_object(file_path=transformer_path,obj=transformer)
                    save_object(file_path=model_path,obj=model)
                    save_object(file_path=target_encoder_path,obj=target_encoder)
                    model_pusher_artifact = ModelPushArtifact(pusher_model_dir=self.model_pusher_config.push_model_dir,
                    saved_model_dir=self.model_pusher_config.saved_model_dir)
                    logging.info(f"Model pusher artifact: {model_pusher_artifact}")
                    return model_pusher_artifact
                except Exception as e:
                    raise SensorException(e,sys)