import sys

from us_visa_approval_prediction.exception import USvisaException
from us_visa_approval_prediction.logger import logging

import os
from us_visa_approval_prediction.constants import DB_NAME, CONNECTION_URL
import pymongo
import certifi

ca = certifi.where()

class MongoDBClient:
    """
    Class Name :   export_data_into_feature_store
    Description :   This method exports the dataframe from mongodb feature store as dataframe
    Output      :   connection to mongodb database
    On Failure  :   raises an exception
    """
    client = None

    def __init__(self, database_name=DB_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = CONNECTION_URL
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {CONNECTION_URL} is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection succesfull")
        except Exception as e:
            raise USvisaException(e,sys)