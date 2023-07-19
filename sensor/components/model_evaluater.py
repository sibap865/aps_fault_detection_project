import pandas as pd
from sensor.predicter import ModelResolver
from sensor.entity import config_entity,artifacts_entity
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils import load_object
from sensor.config import TARGET_COLUMN
import sys
from sklearn.metrics import f1_score
class ModelEvaluater:

    def __init__(self,
                 model_evalution_config:config_entity.ModelEvalutionConfig,
                 data_ingestion_artifact:artifacts_entity.DataIngestionArtifact,
                 data_transformation_artifact:artifacts_entity.DataTransformationArtifact,
                 model_trainer_artifact:artifacts_entity.ModelTrainerArtifact):
        
        try:
            logging.info(f"{'>>'*20} Model Evaluation {'<<'*20}")
            self.model_evalution_config =model_evalution_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver= ModelResolver()
        except Exception as e:
            raise SensorException(e,sys)
    
    def initiate_model_evalution(self) -> artifacts_entity.ModelEvalutionArtifact:
        try:
            #  if saved model dir has models we need to compare
            # which model is best trained or model from saved folder
            logging.info("if saved model dir has models we need to compare\nwhich model is best trained or model from saved folder ")

            latest_dir_path =self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:
                model_eval_artifact =artifacts_entity.ModelEvalutionArtifact(is_model_accepted=True,improved_accuracy=None)
                logging.info(f"model evaluation artifact :{model_eval_artifact}")
                return model_eval_artifact
            
            # finding location of transformer model and target encoder
            logging.info("finding location of transformer model and target encoder")
            transformer_path =self.model_resolver.get_latest_transformer_path()
            model_path =self.model_resolver.get_latest_model_path()
            target_encoder_path =self.model_resolver.get_latest_target_encoder_path()

            # loading object
            logging.info("loading object")
            transformer =load_object(file_path=transformer_path)
            model =load_object(file_path=model_path)
            target_encoder =load_object(file_path=target_encoder_path)

            # current trained model objects
            logging.info("current trained model objects")
            current_transformer=load_object(file_path=self.data_transformation_artifact.transformation_oject_path)
            current_model=load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder=load_object(file_path=self.data_transformation_artifact.target_encoder_path)


            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df =test_df[TARGET_COLUMN]
            
            # acccuracy using previously trained model
            logging.info("acccuracy using previously trained model")
            y_true=target_encoder.transform(target_df)

            input_feature_name =list(transformer.feature_names_in_)


            input_arr =transformer.transform(test_df[input_feature_name])
            y_pred =model.predict(input_arr)
            print(f"Prediction using previous model :{target_encoder.inverse_transform(y_pred[:5])}")
            logging.info(f"Prediction using previous model :{target_encoder.inverse_transform(y_pred[:5])}")
            previous_model_score= f1_score(y_true=y_true,y_pred=y_pred)
            logging.info(f"previous model score :{previous_model_score}")

            y_true=current_target_encoder.transform(target_df)

            input_feature_name=list(current_transformer.feature_names_in_)
            input_arr =current_transformer.transform(test_df[input_feature_name])
            y_pred =current_model.predict(input_arr)

            print(f"Prediction using current model :{current_target_encoder.inverse_transform(y_pred[:5])}")
            logging.info(f"Prediction using current model :{current_target_encoder.inverse_transform(y_pred[:5])}")
            current_model_score= f1_score(y_true=y_true,y_pred=y_pred)
            if current_model_score <= previous_model_score:
                logging.info(f"current model score :{current_model} and previous {previous_model_score}")
                raise Exception("Current trained model is not better then previous one")
            logging.info(f"current model score :{current_model}")
            model_eval_artifact=artifacts_entity.ModelEvalutionArtifact(is_model_accepted=True,improved_accuracy=(current_model_score-previous_model_score))
            logging.info(f"Model evalution artifact : {model_eval_artifact}")

            

        except Exception as e:
            raise SensorException(e,sys)