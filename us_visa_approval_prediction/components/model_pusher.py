import sys

from us_visa_approval_prediction.cloud_storage.gdrive_storage import GoogleDriveStorageService  # Updated import
from us_visa_approval_prediction.exception import USvisaException
from us_visa_approval_prediction.logger import logging
from us_visa_approval_prediction.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from us_visa_approval_prediction.entity.config_entity import ModelPusherConfig
from us_visa_approval_prediction.entity.gdrive_estimator import USvisaEstimator  # Updated import


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.gdrive = GoogleDriveStorageService(folder_name=model_pusher_config.folder_name)  # Updated
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.usvisa_estimator = USvisaEstimator(folder_name=model_pusher_config.folder_name,  # Updated
                                model_path=model_pusher_config.gdrive_model_key_path)  # Updated

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This function is used to initiate all steps of the model pusher

        Output      :   Returns model pusher artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            logging.info("Uploading model to Google Drive folder")

            self.usvisa_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)

            model_pusher_artifact = ModelPusherArtifact(folder_name=self.model_pusher_config.folder_name,  # Updated
                                                        gdrive_model_path=self.model_pusher_config.gdrive_model_key_path)  # Updated

            logging.info("Uploaded model to Google Drive folder")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e