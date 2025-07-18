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

# FastAPI 实例
app = FastAPI(title="Customer Churn Prediction API")

# 允许前端跨域访问（GitHub Pages）
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

# 模型路径配置（保持 models 文件夹不变）
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

    # 读取 CSV 文件
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:
        return {"error": f"Failed to read uploaded file: {str(e)}"}

    # 清洗 + 特征工程（如你已有这些模块）
    try:
        df = load_data(df)
        df = build_feature_pipeline(df)
    except Exception as e:
        return {"error": f"Preprocessing failed: {str(e)}"}

    # 载入模型
    try:
        model_path = MODEL_PATHS[model_type]
        model = joblib.load(model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

    # 预测
    try:
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
