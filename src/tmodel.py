from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

def get_model(config: dict):
    model_type = config.get("type", "xgboost")
    
    if model_type == "xgboost":
        return XGBClassifier(**config.get("params", {}))
    
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(**config.get("params", {}))
    
    else:
        raise ValueError(f"❌ Unknown model type: {model_type}")
