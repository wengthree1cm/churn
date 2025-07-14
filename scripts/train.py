import os
import sys
import yaml
import shap
import joblib
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    RocCurveDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier

# === Add src to path ===
sys.path.append(os.path.abspath("src"))
from data_processing import load_data
from feature_engineering import build_feature_pipeline
from model import get_model
from logs import setup_logger

# === Hyperparameter grids for GridSearchCV ===
param_grids = {
    "xgboost": {
        "n_estimators": [50, 100],
        "max_depth": [3, 5]
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10]
    },
    "logistic": {
        "C": [0.1, 1.0, 10.0]
    }
}

# === LightGBM Optuna optimization ===
def optimize_lgbm(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, pos_label=1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_params
    best_model = LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)
    return best_model, best_params

# === Plotting ===
def plot_and_save_confusion_matrix(model, X_test, y_test, save_path):
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def plot_and_save_shap_summary(model, X, model_type, save_path):
    try:
        if model_type in ["xgboost", "random_forest", "lightgbm"]:
            explainer = shap.Explainer(model, X)
        elif model_type == "logistic":
            explainer = shap.Explainer(model.predict, X)  # fallback
        else:
            print(f"⚠️ SHAP not supported for model: {model_type}")
            return

        shap_values = explainer(X)
        shap.plots.beeswarm(shap_values, show=False)
        plt.title(f"SHAP Summary ({model_type})")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"⚠️ SHAP failed for {model_type}: {e}")


def plot_and_save_roc(model, X_test, y_test, save_path):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title("ROC Curve")
    plt.savefig(save_path)
    plt.close()

def plot_and_save_pr(model, X_test, y_test, save_path):
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title("Precision-Recall Curve")
    plt.savefig(save_path)
    plt.close()

# === Main pipeline ===
def main():
    # Load config
    config_path = "config/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger = setup_logger()
    logger.info("🚀 Training pipeline started")

    # Load data
    df = load_data(config["data_path"])
    X, y = build_feature_pipeline(df, config["target_col"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_seed"]
    )

    # Setup timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Loop through models
    summary = []
    for model_type in ["xgboost", "random_forest", "logistic", "lightgbm"]:
        logger.info(f"🔍 Training model: {model_type}")

        if model_type == "lightgbm":
            model, best_params = optimize_lgbm(X_train, y_train, X_test, y_test)
        else:
            base_model = get_model(model_type)
            param_grid = param_grids.get(model_type, {})
            model_cv = GridSearchCV(base_model, param_grid, cv=3, scoring="f1", n_jobs=-1)
            model_cv.fit(X_train, y_train)
            model = model_cv.best_estimator_
            best_params = model_cv.best_params_

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/model_{model_type}_{timestamp}.joblib"
        joblib.dump(model, model_path)

        # Save reports
        report_dir = os.path.join("reports", timestamp, model_type)
        os.makedirs(report_dir, exist_ok=True)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        with open(os.path.join(report_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        plot_and_save_confusion_matrix(model, X_test, y_test, os.path.join(report_dir, "confusion_matrix.png"))
        plot_and_save_roc(model, X_test, y_test, os.path.join(report_dir, "roc_curve.png"))
        plot_and_save_pr(model, X_test, y_test, os.path.join(report_dir, "precision_recall.png"))
        plot_and_save_shap_summary(model, X_test, model_type, os.path.join(report_dir, "shap_summary.png"))

    logger.info("🏁 Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
