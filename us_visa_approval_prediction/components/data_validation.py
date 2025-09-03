import json
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from pandas import DataFrame
import re

from us_visa_approval_prediction.exception import USvisaException
from us_visa_approval_prediction.logger import logging
from us_visa_approval_prediction.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa_approval_prediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from us_visa_approval_prediction.entity.config_entity import DataValidationConfig
from us_visa_approval_prediction.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config =read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e,sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise USvisaException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")


            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise USvisaException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Method Name :   detect_dataset_drift
        Description :   This method validates if drift is detected using Evidently's built-in logic

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            # Create a report with data drift preset
            report = Report(metrics=[DataDriftPreset()])
            report = report.run(reference_data=reference_df, current_data=current_df)

            # Export to JSON string
            report_json_str = report.json()

            # Parse JSON string to dictionary
            report_json = json.loads(report_json_str)

            # Save the JSON report
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=report_json_str)

            # Extract drift information from the DriftedColumnsCount metric
            metrics = report_json.get("metrics", [])

            # Find the DriftedColumnsCount metric
            for metric in metrics:
                if "DriftedColumnsCount" in metric.get("metric_id", ""):
                    drift_share = metric["value"]["share"]
                    n_drifted_features = int(metric["value"]["count"])

                    # Extract Evidently's threshold from the metric_id
                    # metric_id format: "DriftedColumnsCount(drift_share=0.5)"
                    threshold_match = re.search(r'drift_share=([\d.]+)', metric["metric_id"])
                    evidently_threshold = float(threshold_match.group(1)) if threshold_match else 0.5

                    # Use Evidently's built-in decision logic
                    drift_status = drift_share >= evidently_threshold

                    # Calculate total features for logging
                    value_drift_metrics = [m for m in metrics if "ValueDrift" in m.get("metric_id", "")]
                    n_features = len(value_drift_metrics)

                    logging.info(f"{n_drifted_features}/{n_features} features drifted (drift share: {drift_share:.2%}, threshold: {evidently_threshold})")
                    return drift_status

            # Fallback if DriftedColumnsCount metric is not found
            logging.warning("DriftedColumnsCount metric not found, assuming no drift")
            return False

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))

            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.validate_number_of_columns(dataframe=test_df)

            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            status = self.is_column_exist(df=train_df)

            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.is_column_exist(df=test_df)

            if not status:
                validation_error_msg += f"columns are missing in test dataframe."

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info(f"Drift detected.")
                    validation_error_msg = "Drift detected"
                else:
                    validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation_error: {validation_error_msg}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e