from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

def evaluate_model(pipeline, X_val, y_val):
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    print("📊 Classification Report:")
    print(classification_report(y_val, y_pred))

    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    auc = roc_auc_score(y_val, y_prob)
    print(f"🔥 ROC AUC Score: {auc:.4f}")

    RocCurveDisplay.from_predictions(y_val, y_prob)
    plt.title("ROC Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
