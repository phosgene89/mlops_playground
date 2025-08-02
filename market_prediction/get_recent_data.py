import requests
import json
from datetime import datetime, timedelta

import pandas as pd

TABLE_NAME = "AAPL_market_price"
SYMBOL = "AAPL"

# Load the JSON file
with open("credentials.json", "r") as f:
    credentials = json.load(f)

url = credentials["api"]["endpoint"]

today = datetime.today().date()
one_week_ago = today - timedelta(days=7)

params = {
    "access_key": credentials["api"]["access_key"],
    "symbols": SYMBOL,
    "date_from": one_week_ago.strftime("%Y-%m-%d"),
    "date_to": today.strftime("%Y-%m-%d")
}

# Make the GET request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Failed to retrieve data: {response.status_code} - {response.text}")

df = pd.DataFrame(data['data'])

df_spark = spark.createDataFrame(df)
(df_spark.write.mode("overwrite")
                        .option("overwriteSchema", "true")
                        .saveAsTable(TABLE_NAME))