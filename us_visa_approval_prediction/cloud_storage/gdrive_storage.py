import os
import sys
import pickle
from io import StringIO, BytesIO
from typing import Union, List
import pandas as pd
from pandas import DataFrame, read_csv

from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from us_visa_approval_prediction.configuration.gdrive_connection import GoogleDriveClient
from us_visa_approval_prediction.logger import logging
from us_visa_approval_prediction.exception import USvisaException


class GoogleDriveStorageService:
    """
    Google Drive Storage Service class to replace AWS S3 functionality
    This class provides similar methods to interact with Google Drive instead of S3
    """

    def __init__(self, folder_name: str = 'Visa Approval ML Project'):
        """
        Initialize Google Drive service with authentication using GoogleDriveClient
        
        Args:
            folder_name (str): Name of the folder where models will be stored
        """
        self.folder_name = folder_name
        gdrive_client = GoogleDriveClient()
        self.gdrive_service = gdrive_client.gdrive_service
        self.folder_id = self._get_or_create_folder(folder_name)

    def _get_or_create_folder(self, folder_name: str) -> str:
        """
        Get existing folder or create new folder in Google Drive
        
        Args:
            folder_name (str): Name of the folder to get or create
            
        Returns:
            str: Folder ID in Google Drive
        """
        try:
            # Search for existing folder
            results = self.gdrive_service.files().list(
                q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
                spaces='drive'
            ).execute()
            
            items = results.get('files', [])
            
            if items:
                folder_id = items[0]['id']
                logging.info(f"Found existing folder '{folder_name}' with ID: {folder_id}")
                return folder_id
            else:
                # Create new folder
                folder_metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = self.gdrive_service.files().create(body=folder_metadata, fields='id').execute()
                folder_id = folder.get('id')
                logging.info(f"Created new folder '{folder_name}' with ID: {folder_id}")
                return folder_id
                
        except Exception as e:
            raise USvisaException(f"Failed to get or create folder: {e}", sys)

    def gdrive_file_exists(self, file_name: str) -> bool:
        """
        Check if file exists in Google Drive folder (equivalent to s3_key_path_available)
        
        Args:
            file_name (str): Name of the file to check
            
        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            results = self.gdrive_service.files().list(
                q=f"name='{file_name}' and '{self.folder_id}' in parents and trashed=false",
                spaces='drive'
            ).execute()
            
            items = results.get('files', [])
            return len(items) > 0
            
        except Exception as e:
            raise USvisaException(f"Error checking file existence: {e}", sys)

    @staticmethod
    def read_object(file_content: bytes, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str]:
        """
        Method to read file content with specified format
        
        Args:
            file_content (bytes): Raw file content
            decode (bool): Whether to decode bytes to string
            make_readable (bool): Whether to convert to StringIO object
            
        Returns:
            Union[StringIO, str]: Processed file content
        """
        logging.info("Entered the read_object method of GoogleDriveStorageService class")

        try:
            func = lambda: file_content.decode() if decode is True else file_content
            conv_func = lambda: StringIO(func()) if make_readable is True else func()
            logging.info("Exited the read_object method of GoogleDriveStorageService class")
            return conv_func()

        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_file_object(self, filename: str) -> Union[List[dict], dict]:
        """
        Get file object(s) from Google Drive folder based on filename
        
        Args:
            filename (str): Name of the file to retrieve
            
        Returns:
            Union[List[dict], dict]: File object(s) from Google Drive
        """
        logging.info("Entered the get_file_object method of GoogleDriveStorageService class")

        try:
            results = self.gdrive_service.files().list(
                q=f"name contains '{filename}' and '{self.folder_id}' in parents and trashed=false",
                spaces='drive'
            ).execute()
            
            file_objects = results.get('files', [])
            
            func = lambda x: x[0] if len(x) == 1 else x
            file_objs = func(file_objects)
            
            logging.info("Exited the get_file_object method of GoogleDriveStorageService class")
            return file_objs

        except Exception as e:
            raise USvisaException(e, sys) from e

    def load_model(self, model_name: str, folder_name: str = None, model_dir: str = None) -> object:
        """
        Load model from Google Drive (equivalent to S3 load_model)
        
        Args:
            model_name (str): Name of the model file
            folder_name (str): Not used in Google Drive (kept for compatibility)
            model_dir (str): Directory path within the main folder (optional)
            
        Returns:
            object: Loaded model object
        """
        logging.info("Entered the load_model method of GoogleDriveStorageService class")

        try:
            func = (
                lambda: model_name
                if model_dir is None
                else model_dir + "/" + model_name
            )
            model_file = func()
            
            # Get file from Google Drive
            results = self.gdrive_service.files().list(
                q=f"name='{model_file}' and '{self.folder_id}' in parents and trashed=false",
                spaces='drive'
            ).execute()
            
            items = results.get('files', [])
            if not items:
                raise USvisaException(f"Model file '{model_file}' not found in Google Drive", sys)
            
            file_id = items[0]['id']
            
            # Download file content
            request = self.gdrive_service.files().get_media(fileId=file_id)
            file_content = BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            file_content.seek(0)
            model_obj = file_content.read()
            model = pickle.loads(model_obj)
            
            logging.info("Exited the load_model method of GoogleDriveStorageService class")
            return model

        except Exception as e:
            raise USvisaException(e, sys) from e

    def create_folder(self, folder_name: str) -> str:
        """
        Create a folder in Google Drive
        
        Args:
            folder_name (str): Name of the folder to create
            
        Returns:
            str: Folder ID of created folder
        """
        logging.info("Entered the create_folder method of GoogleDriveStorageService class")

        try:
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [self.folder_id]
            }
            folder = self.gdrive_service.files().create(body=folder_metadata, fields='id').execute()
            folder_id = folder.get('id')
            
            logging.info(f"Created folder '{folder_name}' with ID: {folder_id}")
            logging.info("Exited the create_folder method of GoogleDriveStorageService class")
            return folder_id

        except Exception as e:
            raise USvisaException(e, sys) from e

    def upload_file(self, from_filename: str, to_filename: str, folder_name: str = None, remove: bool = True):
        """
        Upload file to Google Drive (equivalent to S3 upload_file)
        
        Args:
            from_filename (str): Local file path to upload
            to_filename (str): Name for the file in Google Drive
            folder_name (str): Not used in Google Drive (kept for compatibility)
            remove (bool): Whether to remove local file after upload
        """
        logging.info("Entered the upload_file method of GoogleDriveStorageService class")

        try:
            logging.info(f"Uploading {from_filename} file as {to_filename} to Google Drive folder")
            
            # Check if file already exists and delete it
            existing_files = self.gdrive_service.files().list(
                q=f"name='{to_filename}' and '{self.folder_id}' in parents and trashed=false",
                spaces='drive'
            ).execute()
            
            for file in existing_files.get('files', []):
                self.gdrive_service.files().delete(fileId=file['id']).execute()
            
            # Upload new file
            file_metadata = {
                'name': to_filename,
                'parents': [self.folder_id]
            }
            
            media = MediaFileUpload(from_filename, resumable=True)
            file = self.gdrive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            logging.info(f"Uploaded {from_filename} as {to_filename} to Google Drive with ID: {file.get('id')}")

            if remove is True:
                os.remove(from_filename)
                logging.info(f"Remove is set to {remove}, deleted the local file")
            else:
                logging.info(f"Remove is set to {remove}, not deleted the local file")

            logging.info("Exited the upload_file method of GoogleDriveStorageService class")

        except Exception as e:
            raise USvisaException(e, sys) from e

    def upload_df_as_csv(self, data_frame: DataFrame, local_filename: str, drive_filename: str, folder_name: str = None) -> None:
        """
        Upload DataFrame as CSV file to Google Drive (equivalent to S3 upload_df_as_csv)
        
        Args:
            data_frame (DataFrame): DataFrame to upload
            local_filename (str): Local filename for temporary CSV
            drive_filename (str): Filename for Google Drive
            folder_name (str): Not used in Google Drive (kept for compatibility)
        """
        logging.info("Entered the upload_df_as_csv method of GoogleDriveStorageService class")

        try:
            data_frame.to_csv(local_filename, index=None, header=True)
            self.upload_file(local_filename, drive_filename, folder_name)
            logging.info("Exited the upload_df_as_csv method of GoogleDriveStorageService class")

        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_df_from_object(self, file_object: dict) -> DataFrame:
        """
        Get DataFrame from Google Drive file object
        
        Args:
            file_object (dict): Google Drive file object
            
        Returns:
            DataFrame: Pandas DataFrame from CSV content
        """
        logging.info("Entered the get_df_from_object method of GoogleDriveStorageService class")

        try:
            file_id = file_object['id']
            
            # Download file content
            request = self.gdrive_service.files().get_media(fileId=file_id)
            file_content = BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            file_content.seek(0)
            content = self.read_object(file_content.read(), make_readable=True)
            df = read_csv(content, na_values="na")
            
            logging.info("Exited the get_df_from_object method of GoogleDriveStorageService class")
            return df
            
        except Exception as e:
            raise USvisaException(e, sys) from e

    def read_csv(self, filename: str, folder_name: str = None) -> DataFrame:
        """
        Read CSV file from Google Drive and return as DataFrame (equivalent to S3 read_csv)
        
        Args:
            filename (str): Name of the CSV file in Google Drive
            folder_name (str): Not used in Google Drive (kept for compatibility)
            
        Returns:
            DataFrame: Pandas DataFrame from CSV file
        """
        logging.info("Entered the read_csv method of GoogleDriveStorageService class")

        try:
            csv_obj = self.get_file_object(filename)
            df = self.get_df_from_object(csv_obj)
            logging.info("Exited the read_csv method of GoogleDriveStorageService class")
            return df
            
        except Exception as e:
            raise USvisaException(e, sys) from e