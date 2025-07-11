import os
import sys
import yaml
import joblib
from datetime import datetime

sys.path.append(os.path.abspath("src"))

from data_processing import load_data
from feature_engineering import build_feature_pipeline
from model import get_model
from evaluation import evaluate_model
from logs import setup_logger

from sklearn.model_selection import train_test_split

def main():
    # === Load config ===
    config_path = "config/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # === Setup logger ===
    logger = setup_logger()
    logger.info("🚀 Training pipeline started")

    # === Load data ===
    df = load_data(config["data_path"])
    logger.info(f"✅ Data loaded. Shape: {df.shape}")

    # === Feature processing ===
    X, y = build_feature_pipeline(df, target_col=config["target_col"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_seed"]
    )
    logger.info(f"✅ Data split: Train={X_train.shape}, Test={X_test.shape}")

    # === Train model ===
    model = get_model(config["model_type"])
    model.fit(X_train, y_train)
    logger.info(f"✅ Model ({config['model_type']}) trained")

    # === Save model ===
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/model_{config['model_type']}_{timestamp}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"✅ Model saved to: {model_path}")

    # === Evaluate model ===
    evaluate_model(model, X_test, y_test)
    logger.info("✅ Evaluation complete. Reports saved to 'reports/' folder.")

    logger.info("🏁 Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
