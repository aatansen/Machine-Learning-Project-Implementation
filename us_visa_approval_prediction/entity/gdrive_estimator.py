from us_visa_approval_prediction.cloud_storage.gdrive_storage import GoogleDriveStorageService
from us_visa_approval_prediction.exception import USvisaException
from us_visa_approval_prediction.entity.estimator import USvisaModel
import sys
from pandas import DataFrame


class USvisaEstimator:
    """
    This class is used to save and retrieve us_visas model in Google Drive and to do prediction
    """

    def __init__(self, folder_name: str, model_path: str):
        """
        :param folder_name: Name of your model folder in Google Drive
        :param model_path: Name of your model file in the folder
        """
        self.folder_name = folder_name
        self.gdrive = GoogleDriveStorageService(folder_name=folder_name)
        self.model_path = model_path
        self.loaded_model: USvisaModel = None

    def is_model_present(self, model_path: str) -> bool:
        """
        Check if model exists in Google Drive folder

        Args:
            model_path (str): Name of the model file

        Returns:
            bool: True if model exists, False otherwise
        """
        try:
            return self.gdrive.gdrive_file_exists(file_name=model_path)
        except USvisaException as e:
            print(e)
            return False

    def load_model(self) -> USvisaModel:
        """
        Load the model from Google Drive

        Returns:
            USvisaModel: Loaded model object
        """
        return self.gdrive.load_model(self.model_path)

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """
        Save the model to Google Drive

        Args:
            from_file (str): Your local system model path
            remove (bool): Whether to remove local file after upload
        """
        try:
            self.gdrive.upload_file(from_file,
                                   to_filename=self.model_path,
                                   remove=remove)
        except Exception as e:
            raise USvisaException(e, sys)

    def predict(self, dataframe: DataFrame):
        """
        Make predictions using the loaded model

        Args:
            dataframe (DataFrame): Input data for prediction

        Returns:
            Predictions from the model
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise USvisaException(e, sys)