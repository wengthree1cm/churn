import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# 模型列表
MODEL_OPTIONS = {
    "Logistic Regression": "logistic",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm"
}

# 模型文件夹路径
MODEL_DIR = "models"

# 找到最新的模型文件（基于时间戳）
def get_latest_model(model_type):
    files = [f for f in os.listdir(MODEL_DIR) if model_type in f and f.endswith(".joblib")]
    if not files:
        return None
    latest = sorted(files)[-1]
    return os.path.join(MODEL_DIR, latest)

# 页面设置
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📈 Customer Churn Prediction Dashboard")

# 选择模型
model_display = st.selectbox("请选择模型：", list(MODEL_OPTIONS.keys()))
model_type = MODEL_OPTIONS[model_display]
model_path = get_latest_model(model_type)

if not model_path:
    st.warning(f"❌ 没有找到模型 `{model_type}`，请先运行训练脚本。")
    st.stop()

# 加载模型
model = joblib.load(model_path)
st.success(f"✅ 已加载模型：{os.path.basename(model_path)}")

# 上传 CSV
uploaded_file = st.file_uploader("请上传测试数据 CSV 文件（字段需与训练数据一致，去除 label 和 customerID）", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # 尝试提取模型中训练的列名（必须是用 ColumnTransformer/OneHot 后的特征名）
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
        else:
            expected_cols = df.columns.tolist()  # fallback

        # 校验列一致性
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ 缺少以下字段：{missing_cols}")
        else:
            st.success("✅ 数据格式校验通过！开始预测...")

            # 预测
            y_prob = model.predict_proba(df)[:, 1]
            y_pred = model.predict(df)

            result_df = df.copy()
            result_df["Predicted_Probability"] = y_prob
            result_df["Predicted_Label"] = y_pred

            st.subheader("📋 预测结果")
            st.dataframe(result_df.head(20))

            csv_download = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 下载预测结果 CSV", data=csv_download, file_name="churn_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"❌ 读取文件失败：{e}")
