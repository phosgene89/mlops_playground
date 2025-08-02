def load_adj_close_from_spark_table(spark, table_name="AAPL_market_price", return_type="numpy"):
    """
    Loads adj_close column from a Spark table.

    Parameters:
        spark: SparkSession
        table_name: str - name of the table in Databricks
        return_type: 'numpy', 'series', or 'dataframe'

    Returns:
        np.ndarray, pd.Series, or pd.DataFrame
    """
    df_spark = spark.table(table_name).select("date", "adj_close")
    df_pandas = df_spark.toPandas().sort_values("date")

    if return_type == "numpy":
        return df_pandas["adj_close"].values
    elif return_type == "series":
        return df_pandas.set_index("date")["adj_close"]
    elif return_type == "dataframe":
        return df_pandas.set_index("date")
    else:
        raise ValueError(f"Invalid return_type: {return_type}")