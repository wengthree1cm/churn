import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import joblib

from feature_engineering import build_feature_pipeline
from evaluation import evaluate_model

def train_and_save_model(df: pd.DataFrame, label_col: str, model_path: str = "model.joblib"):
    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = build_feature_pipeline(X_train)

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.85,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    evaluate_model(pipeline, X_val, y_val)

    joblib.dump(pipeline, model_path)
    print(f"✅ Model saved to {model_path}")
