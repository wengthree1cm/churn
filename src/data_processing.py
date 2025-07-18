import pandas as pd

def load_data(df):
    df = df.copy()
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
    return df
