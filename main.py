import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import io

from data_processing import load_data
from feature_engineering import build_feature_pipeline

app = FastAPI(title="Customer Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://wengthree1cm.github.io",
        "https://wengthree1cm.github.io/churn"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATHS = {
    "xgboost": "models/model_xgboost.joblib",
    "random_forest": "models/model_random_forest.joblib",
    "logistic": "models/model_logistic.joblib",
    "lightgbm": "models/model_lightgbm.joblib"
}

@app.get("/")
def read_root():
    return {"message": "Welcome to Customer Churn Prediction API!"}

@app.post("/predict")
async def predict_churn(file: UploadFile = File(...), model_type: str = Form(...)):
    model_type = model_type.lower()
    if model_type not in MODEL_PATHS:
        return {"error": f"Invalid model type: {model_type}"}

    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:
        return {"error": f"Failed to read uploaded file: {str(e)}"}

    try:
        df = load_data(df)
        X, _ = build_feature_pipeline(df, target_col="Churn")
    except Exception as e:
        return {"error": f"Preprocessing failed: {str(e)}"}

    try:
        model = joblib.load(MODEL_PATHS[model_type])
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

    try:
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
