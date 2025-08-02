import numpy as np
import pandas as pd
from datetime import datetime

def format_date(input_date: str):
    # Parse the input date (assuming YYYY-MM-DD format)
    date_obj = datetime.strptime(input_date, "%Y-%m-%d")

    # Format the date to the desired output (YYYY-MM-DDTHH:MM:SS+0000)
    output_date = date_obj.strftime("%Y-%m-%dT%H:%M:%S+0000")

    return output_date

# Get historical AAPL data, format it to be consistent with the streaming data, then
# store it in a table.

historical_data = pd.read_csv("historical_data/historical_AAPL_price.csv")
historical_data.columns = ["date", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]

historical_data.date = historical_data.date.apply(lambda x: format_date(x))

REQUIRED_COLS = ['open', 'high', 'low', 'close', 'volume', 'adj_high', 'adj_low',
                'adj_close', 'adj_open', 'adj_volume', 'split_factor', 'dividend',
                'name', 'exchange_code', 'asset_type', 'price_currency', 'symbol',
                'exchange', 'date']

for col in REQUIRED_COLS:
    columns = historical_data.columns
    
    if col not in columns:
        historical_data[col] = np.nan

    if col == 'name':
        historical_data[col] = "Apple Inc"
    elif col == 'exchange_code':
        historical_data[col] = "NASDAQ"
    elif col == 'asset_type':
        historical_data[col] = "stock"
    elif col == 'price_currency':
        historical_data[col] = "USD"
    elif col == 'symbol':
        historical_data[col] = "AAPL"
    elif col == 'exchange':
        historical_data[col] = "XNAS"

historical_data.head()

df_spark = spark.createDataFrame(historical_data)
(df_spark.write.mode("append")
                        .saveAsTable("AAPL_market_price"))