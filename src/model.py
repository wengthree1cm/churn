from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

def get_model(model_type="xgboost", debug_mode=False):
    if model_type == "xgboost":
        return XGBClassifier(
            eval_metric="logloss",
            n_estimators=10 if debug_mode else 100,
            max_depth=2 if debug_mode else 6,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=20 if debug_mode else 100,
            max_depth=5 if debug_mode else None,
            random_state=42
        )
    elif model_type == "logistic":
        return LogisticRegression(
            max_iter=20 if debug_mode else 1000,
            solver="lbfgs"
        )
    elif model_type == "lightgbm":
        return LGBMClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
