import sys
import os  # Added
import json  # Added
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from neuro_mf import ModelFactory
import pickle

from us_visa_approval_prediction.exception import USvisaException
from us_visa_approval_prediction.logger import logging
from us_visa_approval_prediction.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object
from us_visa_approval_prediction.entity.config_entity import ModelTrainerConfig
from us_visa_approval_prediction.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from us_visa_approval_prediction.entity.estimator import USvisaModel
from us_visa_approval_prediction.cloud_storage.gdrive_storage import GoogleDriveStorageService

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                model_trainer_config: ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model

        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)

            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(
                X=x_train,y=y_train,base_accuracy=self.model_trainer_config.expected_accuracy
            )
            model_obj = best_model_detail.best_model

            y_pred = model_obj.predict(x_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)  # Add accuracy if updating entity

            return best_model_detail, metric_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e


    def initiate_model_trainer(self, ) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps

        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            best_model_detail ,metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)


            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")

            usvisa_model = USvisaModel(preprocessing_object=preprocessing_obj,
                                        trained_model_object=best_model_detail.best_model)
            logging.info("Created usvisa model object with preprocessor and model")
            logging.info("Created best model file path.")
            save_object(self.model_trainer_config.trained_model_file_path, usvisa_model)

            # Added: Save metrics.json for model report
            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)  # Ensure dir exists
            metrics_path = os.path.join(model_dir, "metrics.json")
            # Re-compute accuracy here if not added to artifact (to avoid entity change)
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            y_pred = best_model_detail.best_model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            metrics_dict = {
                "accuracy": accuracy,
                "f1_score": metric_artifact.f1_score,
                "precision": metric_artifact.precision_score,
                "recall": metric_artifact.recall_score,
                "best_cv_score": best_model_detail.best_score,  # From neuro_mf (assumed CV score)
                "best_parameters": best_model_detail.best_parameters  # From neuro_mf
            }
            with open(metrics_path, "w") as f:
                json.dump(metrics_dict, f)
            logging.info(f"Saved metrics to {metrics_path}")

            # Added: Save feature_names.pkl if extractable from preprocessor
            if hasattr(preprocessing_obj, 'get_feature_names_out'):
                try:
                    feature_names = preprocessing_obj.get_feature_names_out()
                    feature_names_path = os.path.join(model_dir, "feature_names.pkl")
                    with open(feature_names_path, "wb") as f:
                        pickle.dump(feature_names, f)
                    logging.info(f"Saved feature names to {feature_names_path}")
                except Exception as e:
                    logging.info(f"Failed to extract/save feature names: {e}")
            try:
                # instantiate Google Drive service (uses your default folder set in the service)
                gdrive_service = GoogleDriveService = GoogleDriveStorageService(folder_name="Visa Approval ML Project")
                # Upload trained model file (keep same name as in Drive)
                local_model_path = self.model_trainer_config.trained_model_file_path
                model_filename_in_drive = os.path.basename(local_model_path)
                gdrive_service.upload_file(from_filename=local_model_path, to_filename=model_filename_in_drive, remove=False)
                logging.info(f"Uploaded trained model to Google Drive as {model_filename_in_drive}")

                # Upload metrics.json
                metrics_filename_in_drive = os.path.basename(metrics_path)
                gdrive_service.upload_file(from_filename=metrics_path, to_filename=metrics_filename_in_drive, remove=False)
                logging.info(f"Uploaded metrics to Google Drive as {metrics_filename_in_drive}")

                # Upload feature_names.pkl if exists
                if feature_names_path is not None and os.path.exists(feature_names_path):
                    feature_names_filename_in_drive = os.path.basename(feature_names_path)
                    gdrive_service.upload_file(from_filename=feature_names_path, to_filename=feature_names_filename_in_drive, remove=False)
                    logging.info(f"Uploaded feature names to Google Drive as {feature_names_filename_in_drive}")
                else:
                    logging.info("No feature_names.pkl to upload.")
            except Exception as e:
                logging.exception(f"Failed to upload artifacts to Google Drive: {e}")
            # ----------------------------------------------------------------------------------------

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e