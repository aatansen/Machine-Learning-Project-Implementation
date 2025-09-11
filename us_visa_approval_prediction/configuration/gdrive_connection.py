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

        if GoogleDriveClient.gdrive_service is None:

            # Determine token path
            project_root_token = os.path.join(os.getcwd(), "token.pickle")
            if os.path.exists(project_root_token):
                __token_path = project_root_token
            elif token_path is not None:
                __token_path = token_path
            elif GDRIVE_TOKEN_PATH is not None:
                __token_path = GDRIVE_TOKEN_PATH
            else:
                raise Exception("No token file path provided or found in project root.")

            # Load credentials
            try:
                with open(__token_path, 'rb') as token:
                    creds = pickle.load(token)
                    logging.info(f"Token pickle loaded successfully from {__token_path}")
            except Exception as e:
                logging.warning(f"Token pickle load failed from {__token_path}")
                raise Exception(f"Failed to load credentials from {__token_path}: {str(e)}")

            # Validate credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                        logging.info("Credentials refreshed successfully")
                    except Exception as e:
                        raise Exception(f"Failed to refresh credentials: {str(e)}")
                else:
                    raise Exception(f"Invalid credentials in {__token_path}")

            # Build service
            try:
                GoogleDriveClient.gdrive_service = build('drive', 'v3', credentials=creds)
                logging.info("Google Drive service initialized successfully")
            except Exception as e:
                logging.warning("Google Drive service initialization failed")
                raise Exception(f"Failed to build Google Drive service: {str(e)}")

        self.gdrive_service = GoogleDriveClient.gdrive_service
