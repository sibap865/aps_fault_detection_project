import os
import sys
import yaml


from sensor.entity import config_entity, artifacts_entity
from sensor.config import TARGET_COLUMN
from sensor.config import TARGET_FEATURE_MAPING
from sensor.exception import SensorException
from sensor.logger import logging
from sensor import utils

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


class DataTransformation:
    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifacts_entity.DataIngestionArtifact) -> None:
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise SensorException(e, sys)

    @classmethod
    def get_data_transformer_object(cls,) -> Pipeline:
        try:
            # Create a pipeline with simple imputer with strategy constant and fill value 0
            pipeline = Pipeline(steps=[
                ('Imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('RobustScaler', RobustScaler())])

            return pipeline
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_transformation(self) -> artifacts_entity.DataTransformationArtifact:
        try:
            # reading train and test dataframe
            train_df = pd.read_csv(
                self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # selecting input feature for training and test dataframe
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            # selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)
            # transformation on target column
            target_feature_train_arr = label_encoder.transform(
                target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(
                target_feature_test_df)

            transformation_pipeline = DataTransformation.get_data_transformer_object()

            transformation_pipeline.fit(input_feature_train_df)
            # tranforming input feature
            input_feature_train_arr = transformation_pipeline.transform(
                input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(
                input_feature_test_df)
            smt = SMOTETomek(sampling_strategy="minority",random_state=42)
            logging.info(
                f"Before resampling in training set input :{input_feature_train_arr.shape} in test set input :{input_feature_test_arr.shape}")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(
                input_feature_train_arr, target_feature_train_arr)
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(
                input_feature_test_arr, target_feature_test_arr)
            logging.info(
                f"After resampling in training set input :{input_feature_train_arr.shape} in test set input :{input_feature_test_arr.shape}")

            # target encoder saving
            train_arr = np.c_[input_feature_train_arr,
                              target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            # save numpy arr
            utils.save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_path, array=train_arr)
            utils.save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_path, array=test_arr)

            # save pipeline
            utils.save_object(
                file_path=self.data_transformation_config.transformation_oject_path, obj=transformation_pipeline)

            utils.save_object(
                file_path=self.data_transformation_config.target_encoder_path, obj=label_encoder)

            data_transformation_artifact = artifacts_entity.DataTransformationArtifact(
                transformation_oject_path=self.data_transformation_config.transformation_oject_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path,
                target_encoder_path=self.data_transformation_config.target_encoder_path
            )
            logging.info(
                f"data transformation object : {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)