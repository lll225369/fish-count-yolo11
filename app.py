import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# 页面配置
st.set_page_config(page_title="鱼类计数 YOLOv11", page_icon="🐟")
st.title("🐟 基于YOLOv11的鱼类识别与计数小程序")
st.subheader("上传图片，自动检测、统计鱼类数量")

# 加载模型（同目录下best.pt）
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# 上传组件
upload_file = st.file_uploader("上传鱼类图片", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    img = Image.open(upload_file)
    st.image(img, caption="原图", use_column_width=True)

    # 推理
    result = model(img)
    boxes = result[0].boxes
    count = len(boxes)

    # 画框
    res_plot = result[0].plot()
    res_img = Image.fromarray(cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB))

    st.image(res_img, caption=f"检测结果，鱼类总数：{count}", use_column_width=True)
    st.success(f"✅ 检测完成，画面内一共识别到 {count} 条鱼")
