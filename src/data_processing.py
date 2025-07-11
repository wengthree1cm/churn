import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 删除 customerID，如果存在
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    return df
