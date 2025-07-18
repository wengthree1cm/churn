from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
import io

app = FastAPI(title="Customer Churn Prediction API")

# 允许跨域（重要！）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://wengthree1cm.github.io"],  # 或者改成 GitHub Pages 的 URL 更安全
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型路径
MODEL_PATHS = {
    "xgboost": "models/xgboost_model.pkl",
    "random_forest": "models/random_forest_model.pkl",
    "logistic": "models/logistic_regression_model.pkl",
    "lightgbm": "models/lightgbm_model.pkl"
}

@app.get("/")
def read_root():
    return {"message": "Welcome to Customer Churn Prediction API!"}

@app.post("/predict/")
async def predict_churn(file: UploadFile = File(...), model_type: str = Form(...)):
    # 验证模型
    model_type = model_type.lower()
    if model_type not in MODEL_PATHS:
        return {"error": f"Invalid model type: {model_type}"}
    
    # 读取上传的 CSV 文件为 DataFrame
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # 加载模型并预测
    model = joblib.load(MODEL_PATHS[model_type])
    try:
        preds = model.predict(df)
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
    
    # 拼结果
    df["prediction"] = preds
    return df.to_dict(orient="records")
