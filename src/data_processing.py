import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from a CSV file"""
    try:
        data = pd.read_csv(path)
        print(f"✅ Loaded data from {path} with shape {data.shape}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Data file not found at {path}")

if __name__ == "__main__":
    df = load_data("data/hosp_1/data.csv")
    print(df.head())
