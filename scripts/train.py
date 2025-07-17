import os
import sys
import yaml
import shap
import joblib
import optuna
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
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
os.environ["SHAP_PROGRESS_BAR"] = "False"
# === Load config ===
config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

debug_mode = config.get("debug_mode", False)
if config.get("debug_mode", False):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
# === Hyperparameter grids for GridSearchCV ===
param_grids = {
    "xgboost": {
        "n_estimators": [10] if debug_mode else [50, 100],
        "max_depth": [2] if debug_mode else [3, 5]
    },
    "random_forest": {
        "n_estimators": [50] if debug_mode else [100, 200],
        "max_depth": [5] if debug_mode else [None, 10]
    },
    "logistic": {
        "C": [1.0] if debug_mode else [0.1, 1.0, 10.0]
    }
}

# === LightGBM Optuna optimization ===
def optimize_lgbm(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 100) if debug_mode else trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 3) if debug_mode else trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.2) if debug_mode else trial.suggest_float("learning_rate", 0.05, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 20, 30) if debug_mode else trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 20) if debug_mode else trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, pos_label=1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3 if debug_mode else 30)

    best_params = study.best_params
    best_model = LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)
    return best_model, best_params

# === Plotting functions ===
def plot_and_save_confusion_matrix(model, X_test, y_test, save_path):
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()
import shap
import matplotlib.pyplot as plt


import shap
import matplotlib.pyplot as plt

def plot_and_save_shap_summary(model, X, model_type, save_path):
    import logging
    logger = logging.getLogger()
    
    try:
        X = X.toarray() if hasattr(X, "toarray") else X

        if model_type == "random_forest":
            explainer = shap.TreeExplainer(model)
            shap_values_array = explainer.shap_values(X)

            if isinstance(shap_values_array, list):
                shap_values_array = shap_values_array[1]
                base_values = explainer.expected_value[1]
            else:
                base_values = explainer.expected_value

            feature_names = getattr(model, "feature_names_in_", [f"feature_{i}" for i in range(X.shape[1])])

            # ✅ Debug 打印维度
            print(f"[DEBUG] shap_values shape: {shap_values_array.shape}, X shape: {X.shape}")

            # ✅ 构造 Explanation 对象
            shap_values = shap.Explanation(
                values=shap_values_array,
                base_values=base_values,
                data=X,
                feature_names=feature_names
            )

        else:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

        shap.plots.beeswarm(shap_values, show=False)
        plt.title(f"SHAP Summary ({model_type})")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"✅ SHAP summary generated for {model_type}")

    except Exception as e:
        logger.warning(f"⚠️ SHAP failed for {model_type}: {e}")







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
    logger = setup_logger()
    logger.info("🚀 Training pipeline started (debug_mode = %s)", debug_mode)

    df = load_data(config["data_path"])
    X, y = build_feature_pipeline(df, config["target_col"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_seed"]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_to_run = [ "xgboost", "random_forest", "logistic","lightgbm"]

    for model_type in models_to_run:
        logger.info(f"🔍 Training model: {model_type}")

        if model_type == "lightgbm":
            model, best_params = optimize_lgbm(X_train, y_train, X_test, y_test)
        else:
            base_model = get_model(model_type,debug_mode=debug_mode)
            logger.info(f"📌 Model params for {model_type}: {base_model.get_params()}")
            param_grid = param_grids.get(model_type, {})
            model_cv = GridSearchCV(base_model, param_grid, cv=3, scoring="f1", n_jobs=-1)
            model_cv.fit(X_train, y_train)
            model = model_cv.best_estimator_
            best_params = model_cv.best_params_

        # Save model (overwrite with fixed name)
        os.makedirs("models", exist_ok=True)
        model_path = f"models/model_{model_type}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"✅ Saved model: {model_path}")


        # Save reports
        report_dir = os.path.join("reports", timestamp, model_type)
        os.makedirs(report_dir, exist_ok=True)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        with open(os.path.join(report_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        plot_and_save_confusion_matrix(model, X_test, y_test, os.path.join(report_dir, "confusion_matrix.png"))
        plot_and_save_roc(model, X_test, y_test, os.path.join(report_dir, "roc_curve.png"))
        plot_and_save_pr(model, X_test, y_test, os.path.join(report_dir, "precision_recall.png"))
        plot_and_save_shap_summary(model, X_test, model_type, os.path.join(report_dir, "shap_summary.png"))

    logger.info("🏁 Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
