# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Customer Churn Prediction API")

# 加载模型路径
MODEL_PATHS = {
    "xgboost": "models/xgboost_model.pkl",
    "random_forest": "models/random_forest_model.pkl",
    "logistic_regression": "models/logistic_regression_model.pkl",
    "lightgbm": "models/lightgbm_model.pkl"
}

# 定义输入数据
class CustomerInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    model_type: str  # "xgboost", "random_forest", "logistic_regression", "lightgbm"

@app.get("/")
def read_root():
    return {"message": "Welcome to Customer Churn Prediction API!"}

@app.post("/predict/")
def predict_churn(data: CustomerInput):
    model_type = data.model_type.lower()
    if model_type not in MODEL_PATHS:
        return {"error": f"Invalid model type: {model_type}"}
    
    model = joblib.load(MODEL_PATHS[model_type])
    features = [[data.feature1, data.feature2, data.feature3]]
    prediction = model.predict(features)[0]
    return {"prediction": int(prediction)}
