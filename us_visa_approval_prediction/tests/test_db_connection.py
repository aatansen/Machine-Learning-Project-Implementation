import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CONNECTION_URL = os.getenv("CONNECTION_URL")

print("DB_NAME:", DB_NAME)
print("COLLECTION_NAME:", COLLECTION_NAME)
print("CONNECTION_URL:", CONNECTION_URL)
