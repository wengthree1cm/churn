import sys
import os
sys.path.append(os.path.abspath("src"))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import yaml

from src.data_processing import load_data
from src.feature_engineering import build_feature_pipeline
from src.tmodel import get_model

def run_pipeline(config_path: str):
    print("🚀 Pipeline starting...")
    
    # === Step 1: Load config ===
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # === Step 2: Load & split data ===
    df = load_data(config['data']['path'])
    X, y = build_feature_pipeline(df, config['data']['target'])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # === Step 3: Train model ===
    model = get_model(config['model'])
    model.fit(X_train, y_train)

    # === Step 4: Evaluate ===
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred_proba)
    print(f"✅ Validation ROC-AUC: {score:.4f}")

    # === Step 5: Save model ===
    joblib.dump(model, 'models/model.pkl')
    print("✅ Model saved to models/model.pkl")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    run_pipeline(args.config)
