import pymongo
import pandas as pd
import json
# Provide the mongodb localhost url to connect python to mongodb.
client = pymongo.MongoClient("mongodb+srv://apsdata:clone123@cluster0.t3zj5ri.mongodb.net/?retryWrites=true&w=majority")

DATA_FILE_PATH="aps_failure_training_set1.csv"
DATABASE_NAME="aps"
COLLECTION_NAME="sensor"


if __name__=="__main__":
    data =client["aps"]
    data =data["sensor"]
    data =data.find()
    df = pd.read_json(data)
    print(f"Rows and columns: {df.shape}")
    print("done")








