import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

def evaluate_model(model, X_test, y_test, model_name="model", save_path="reports"):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # === 1. Classification Report ===
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "classification_report.json"), "w") as f:
        json.dump(report_dict, f, indent=4)

    with open(os.path.join(save_path, "roc_auc.txt"), "w") as f:
        f.write(f"{model_name} ROC AUC: {roc_auc:.4f}\n")

    # === 2. Plot ROC Curve ===
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, f"roc_curve_{model_name}.png"))
    plt.close()

    return report_dict, roc_auc
