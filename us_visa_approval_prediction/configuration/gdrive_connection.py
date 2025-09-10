import os
import pickle
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from us_visa_approval_prediction.constants import GDRIVE_TOKEN_PATH, GDRIVE_SCOPES
from us_visa_approval_prediction.logger import logging

class GoogleDriveClient:

    gdrive_service = None

    def __init__(self, token_path=None):
        """
        This Class gets google drive credentials from token.pickle file and creates a connection with Google Drive API
        and raise exception when token file is not found or invalid
        """

        if GoogleDriveClient.gdrive_service == None:
            __token_path = GDRIVE_TOKEN_PATH if token_path is None else token_path

            if __token_path is None:
                __token_path = "token.pickle"  # Default path

            if not os.path.exists(__token_path):
                logging.warning(f"Token file not found at: {__token_path}")
                raise Exception(f"Token file not found at: {__token_path}")

            # Load credentials from token.pickle
            try:
                with open(__token_path, 'rb') as token:
                    creds = pickle.load(token)
                    logging.info("Token pickle load succesfull")

            except Exception as e:
                logging.warning("Token pickle load failed")
                raise Exception(f"Failed to load credentials from {__token_path}: {str(e)}")

            # Validate credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                    except Exception as e:
                        raise Exception(f"Failed to refresh credentials: {str(e)}")
                else:
                    raise Exception(f"Invalid credentials in {__token_path}")

            # Create Google Drive service
            try:
                GoogleDriveClient.gdrive_service = build('drive', 'v3', credentials=creds)
                logging.info("Google Drive load succesfull")
            except Exception as e:
                logging.warning("Google Drive service failed")
                raise Exception(f"Failed to build Google Drive service: {str(e)}")

        self.gdrive_service = GoogleDriveClient.gdrive_service