import os
import sys
import yaml
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split

# === Add custom src path ===
sys.path.append(os.path.abspath("src"))

from data_processing import load_data
from feature_engineering import build_feature_pipeline
from model import get_model
from evaluation import evaluate_model
from logs import setup_logger

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_probs, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.plot(recall, precision, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

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

    # === Feature engineering ===
    X, y = build_feature_pipeline(df, config["target_col"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_seed"]
    )
    logger.info(f"✅ Data split: Train={X_train.shape}, Test={X_test.shape}")

    # === Model training ===
    model = get_model(config["model_type"])
    model.fit(X_train, y_train)
    logger.info(f"✅ Model ({config['model_type']}) trained")

    # === Save model ===
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/model_{config['model_type']}_{timestamp}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"✅ Model saved to: {model_path}")

    # === Evaluation ===
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    os.makedirs("reports", exist_ok=True)
    with open(f"reports/classification_report_{timestamp}.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, f"reports/confusion_matrix_{timestamp}.png")
    plot_roc_curve(y_test, y_probs, f"reports/roc_curve_{timestamp}.png")
    plot_precision_recall_curve(y_test, y_probs, f"reports/precision_recall_{timestamp}.png")

    logger.info("✅ Evaluation complete. Charts and report saved in 'reports/'")
    logger.info("🏁 Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
