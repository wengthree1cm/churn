from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
import io
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Customer Churn Prediction API")

# ✅ 修正后的 CORS 设置
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

# 模型路径
MODEL_PATHS = {
    "xgboost": os.path.join(BASE_DIR, "models", "model_xgboost.joblib"),
    "random_forest": os.path.join(BASE_DIR, "models", "model_random_forest.joblib"),
    "logistic": os.path.join(BASE_DIR, "models", "model_logistic.joblib"),
    "lightgbm": os.path.join(BASE_DIR, "models", "model_lightgbm.joblib"),
}

@app.get("/")
def read_root():
    return {"message": "Welcome to Customer Churn Prediction API!"}

@app.post("/predict")
async def predict_churn(file: UploadFile = File(...), model_type: str = Form(...)):
    model_type = model_type.lower()
    if model_type not in MODEL_PATHS:
        return {"error": f"Invalid model type: {model_type}"}

    model_path = MODEL_PATHS[model_type]
    model = joblib.load(model_path)
    df = pd.read_csv(io.BytesIO(await file.read()))
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}
