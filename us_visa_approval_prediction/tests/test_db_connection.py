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

# Insert data to MongoDB
import os
import pandas as pd
import pymongo
import certifi

# Get values from environment
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CONNECTION_URL = os.getenv("CONNECTION_URL")

# Read data
csv_path = os.path.join(BASE_DIR, "us_visa_approval_prediction", "data", "raw", "EasyVisa.csv")
df = pd.read_csv(csv_path)
data = df.to_dict(orient='records')

# Connect to MongoDB
client = pymongo.MongoClient(CONNECTION_URL, tlsCAFile=certifi.where())
data_base = client[DB_NAME]
collection = data_base[COLLECTION_NAME]

# Insert Data
result = collection.insert_many(data)

# Confirmation print
print(f"✅ Successfully inserted {len(result.inserted_ids)} records into '{DB_NAME}.{COLLECTION_NAME}'")

# Optional: Verify first few documents
sample_records = collection.find().limit(3)
print("\n📌 Sample inserted documents:")
for i, record in enumerate(sample_records, start=1):
    print(f"{i}. {record}")

