import os,sys
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.predicter import ModelResolver
import pandas as pd
import numpy as np
from datetime import datetime
from sensor.utils import load_object

PREDICTION_DIR ="predictiion"


def start_batch_prediction(input_file_path:str):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Reading file :{input_file_path}")
        model_resolver =ModelResolver(model_registry="saved_models")
        df =pd.read_csv(input_file_path)

        logging.info("loading transformer to transform dataset")
        transformer =load_object(model_resolver.get_latest_transformer_path())
        input_feature_name =list(transformer.feature_names_in_)
        input_arr =transformer.transform(df[input_feature_name])
        logging.info("loading model to make prediction ")
        model =load_object(model_resolver.get_latest_model_path())
        y_pred=model.predict(input_arr)
        logging.info("target encoder to convert predicted column to categorical")
        target_encoder =load_object(model_resolver.get_latest_target_encoder_path())
        y_pred_cat=target_encoder.inverse_transform(y_pred)
        df["prediction"]=y_pred
        df["cat_pred"]=y_pred_cat
        prediction_file_name=os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path =os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(path_or_buf=prediction_file_path,index=False,header=True)
        return prediction_file_path


    except Exception as e:
        raise SensorException()