from dataclasses import dataclass
@dataclass
class DataIngestionArtifact:
    feature_store_file_path:str
    train_file_path:str
    test_file_path:str
@dataclass
class DataValidationArtifact:
    report_file_path:str
@dataclass
class DataTransformationArtifact:
    transformation_oject_path:str
    transformed_train_path:str
    transformed_test_path:str
    target_encoder_path:str

@dataclass
class ModelTrainerArtifact:
    model_path:str
    f1_train_score:float
    f1_test_score:float

@dataclass
class ModelEvalutionArtifact:
    is_model_accepted:bool
    improved_accuracy:float

@dataclass
class ModelPushArtifact:
    pusher_model_dir:str
    saved_model_dir:str
