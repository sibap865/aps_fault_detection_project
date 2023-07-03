import pymongo
import pandas as pd
import json
# Provide the mongodb localhost url to connect python to mongodb.
client = pymongo.MongoClient("mongodb+srv://apsdata:clone123@cluster0.t3zj5ri.mongodb.net/?retryWrites=true&w=majority")

DATA_FILE_PATH="aps_failure_training_set1.csv"
DATABASE_NAME="aps"
COLLECTION_NAME="sensor"


if __name__=="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    #Convert dataframe to json so that we can dump these record in mongo db
    df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    # print(json_record[1])

    #insert converted json record to mongo db
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
    print("done")








