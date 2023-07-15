import os
import sys
import yaml
from sensor.utils import write_yml_file
from sensor.utils import convert_column_to_float

from sensor.entity import config_entity, artifacts_entity
from sensor.config import TARGET_COLUMN

from sensor.exception import SensorException
from sensor.logger import logging

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Optional


class DataValidation:
    def __init__(self,
                 data_validation_config: config_entity.DataValidationConfig,
                 data_ingestion_artifact: artifacts_entity.DataIngestionArtifact) -> None:
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise SensorException(e, sys)

    def drop_missing_value_column(self, df: pd.DataFrame, report_key_name: str) -> Optional[pd.DataFrame]:
        """
        This function will drop column which contains missing value more then specifid thresold

        df: panadas dataframe
        thresold: percentage criteria to drop columns
        ===============================================================================
        returns: Pandas DataFrame if atleast one column has missing value less then thresold else None
        """
        try:
            thresold = self.data_validation_config.drop_null_value_thresold
            null_col = df.isna().sum()/df.shape[0]
            # selecting columns having null values more then thresold value
            logging.info(
                f"selecting column name which contains null abone to {thresold}")
            drop_columns_name = null_col[null_col > thresold].index
            logging.info(f"columns to drop {list(drop_columns_name)}")
            self.validation_error[report_key_name] = list(drop_columns_name)
            df.drop(drop_columns_name, axis=1, inplace=True)

            # returns None if non of xthe column left after thresolding
            if df.shape[1] == 0:
                return None
            return df
        except Exception as e:
            return SensorException(e, sys)

    def is_required_columns_exist(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str) -> bool:
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns
            logging.info(
                f"columns: current {(current_columns)} and  base col : {base_columns} ")
            missing_col = [
                col for col in base_columns if col not in current_columns]
            logging.info(f"columns: [{missing_col} not available]")
            if len(missing_col) > 0:
                self.validation_error[report_key_name] = missing_col
                return False
            return True
        except Exception as e:
            raise SensorException(e, sys)

    def data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str):
        try:
            drift_report = dict()

            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data, current_data = base_df[base_column], current_df[base_column]
                # Null hypothesis is that both column data drawn from same distrubtion

                logging.info(
                    f"Hypothesis {base_column}: {base_data.dtype}, {current_data.dtype} ")
                same_distribution = ks_2samp(base_data, current_data)

                if same_distribution.pvalue > 0.05:
                    # We are accepting null hypothesis
                    drift_report[base_column] = {
                        "pvalues": float(same_distribution.pvalue),
                        "same_distribution": True
                    }
                else:
                    drift_report[base_column] = {
                        "pvalues": float(same_distribution.pvalue),
                        "same_distribution": False
                    }
                    # different distribution

            self.validation_error[report_key_name] = drift_report
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_validation(self) -> artifacts_entity.DataValidationArtifact:
        try:
            logging.info("reading base DataFrame")

            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            logging.info("replace 'na' values in base DataFrame")
            # base df has nan value as 'na'
            base_df.replace("na", np.nan, inplace=True)
            logging.info(
                f"drop 'null' values columns from base DataFrame {base_df.shape}")
            base_df = self.drop_missing_value_column(
                df=base_df, report_key_name="missing_values_within_base_dset")

            logging.info("Reading train DataFrame")
            train_df = pd.read_csv(
                self.data_ingestion_artifact.train_file_path)
            logging.info("Reading test DataFrame")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_df = self.drop_missing_value_column(
                df=train_df, report_key_name="missing_values_within_train_dset")
            logging.info(
                f"drop 'null' values columns from train DataFrame {train_df.shape}")
            test_df = self.drop_missing_value_column(
                df=test_df, report_key_name="missing_values_within_test_dset")
            logging.info(
                f"drop 'null' values columns from test DataFrame {test_df.shape}")

            exclude_columns = [TARGET_COLUMN]

            base_df = convert_column_to_float(
                df=base_df, exclude_columns=exclude_columns)
            train_df = convert_column_to_float(
                df=train_df, exclude_columns=exclude_columns)
            test_df = convert_column_to_float(
                df=test_df, exclude_columns=exclude_columns)

            logging.info(
                "checking for required column exist in train DataFrame")
            train_df_col_status = self.is_required_columns_exist(
                base_df=base_df, current_df=train_df, report_key_name="missing_columns_within_train_dset")
            logging.info(
                "checking for required column exist in test DataFrame")
            test_df_col_status = self.is_required_columns_exist(
                base_df=base_df, current_df=test_df, report_key_name="missing_columns_within_test_dset")
            self.validation_error["train_df_col_status"] = train_df_col_status
            self.validation_error["test_df_col_status"] = test_df_col_status

            if train_df_col_status:
                logging.info("checking data drift in train DataFrame")
                self.data_drift(base_df=base_df, current_df=train_df,
                                report_key_name="data_drift_within_train_dset")
            if test_df_col_status:
                logging.info("checking data drift in test DataFrame")
                self.data_drift(base_df=base_df, current_df=test_df,
                                report_key_name="data_drift_values_within_train_dset")

            # Write the report in yaml format
            logging.info("Writing report in yaml file ")
            write_yml_file(
                file_path=self.data_validation_config.report_file_path, data=self.validation_error)
            data_validation_artifact = artifacts_entity.DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path)
            logging.info(
                f"Data Validation artifact : {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise SensorException(e, sys)
