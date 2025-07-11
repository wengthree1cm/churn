from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_model(model_type="xgboost"):
    if model_type == "xgboost":
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "logistic":
        return LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
