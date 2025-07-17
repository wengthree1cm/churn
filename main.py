from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
import pandas as pd
import joblib
import os
import io
import sys

# 加入 src 目录到模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# 导入自定义模块
from data_processing import load_data

# 初始化 FastAPI 应用
app = FastAPI(title="Customer Churn Prediction API")

# 允许跨域（方便前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型类型枚举
class ModelType(str, Enum):
    xgboost = "xgboost"
    random_forest = "random_forest"
    logistic = "logistic"
    lightgbm = "lightgbm"

MODEL_DIR = "models"

# 获取模型路径
def get_model_path(model_type: str):
    model_path = os.path.join(MODEL_DIR, f"model_{model_type}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path

# 根路径测试
@app.get("/")
def read_root():
    return {"message": "🎉 Welcome to the Customer Churn Prediction API!"}

# 主预测接口
@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model_type: ModelType = Form(...)
):
    try:
        # 加载模型
        model_path = get_model_path(model_type)
        model = joblib.load(model_path)

        # 读取上传数据
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # 清洗字段
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])

        # 校验列一致性
        expected_cols = list(model.feature_names_in_)
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            return {"error": f"Missing columns: {missing_cols}"}

        # 预测
        y_prob = model.predict_proba(df)[:, 1]
        y_pred = model.predict(df)

        df["Predicted_Probability"] = y_prob
        df["Predicted_Label"] = y_pred

        # 返回前 20 行
        return df.head(20).to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}
