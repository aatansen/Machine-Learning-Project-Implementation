import os
from datetime import date
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CONNECTION_URL = os.getenv("CONNECTION_URL")

PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR: str = "artifact"

TARGET_COLUMN = "case_status"
CURRENT_YEAR = date.today().year
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "usvisa.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join(BASE_DIR,"us_visa_approval_prediction", "config", "schema.yaml")

GDRIVE_TOKEN_PATH = os.path.join(BASE_DIR,"gdrive_setup","token.pickle")
GDRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']
MODEL_FILE_NAME = "model.pkl"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "visa_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join(BASE_DIR,"us_visa_approval_prediction", "config", "model.yaml")


"""
MODEL EVALUATION related constant
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.80
FOLDER_NAME = "Visa Approval ML Project"
MODEL_PUSHER_GDRIVE_KEY = "model-registry"


"""
Web App related constant
"""
APP_HOST = "0.0.0.0"
APP_PORT = 8080