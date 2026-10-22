import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile

# 页面美化配置
st.set_page_config(
    page_title="鱼类识别计数系统 | YOLOv11",
    page_icon="🐟",
    layout="wide"
)

# 自定义css美化
st.markdown("""
<style>
.main {
    background-color: #f0f8ff;
}
h1 {
    color:#006699;
    text-align:center;
}
.stUploader {
    border:2px dashed #0088bb;
    border-radius:12px;
    padding:15px;
}
</style>
""", unsafe_allow_html=True)

# 标题区域
st.markdown("<h1>🐟 基于YOLOv11的水产鱼类识别与计数系统</h1>", unsafe_allow_html=True)
st.divider()
st.markdown("<p style='text-align:center; color:#336688;'>支持图片、短视频输入，自动检测目标鱼类并统计数量</p>", unsafe_allow_html=True)

# 加载模型
@st.cache_resource
def load_model():
    return YOLO("best.pt")
model = load_model()

# 选择模式
mode = st.radio("选择输入类型", ["📷 图片检测", "🎬 视频检测"], horizontal=True)

if mode == "📷 图片检测":
    up_img = st.file_uploader("上传鱼类图片", type=["jpg","jpeg","png"])
    if up_img is not None:
        col1, col2 = st.columns(2)
        img = Image.open(up_img)
        with col1:
            st.image(img, caption="原始图像", use_column_width=True)
        #推理
        res = model(img)
        count = len(res[0].boxes)
        plot_img = res[0].plot()
        plot_img = Image.fromarray(cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB))
        with col2:
            st.image(plot_img, caption=f"检测结果，鱼类总数：{count}", use_column_width=True)
        st.success(f"✅ 检测完成，画面识别到 {count} 条鱼")

elif mode == "🎬 视频检测":
    up_vid = st.file_uploader("上传短视频（建议15s以内）", type=["mp4"])
    if up_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(up_vid.read())
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        bar = st.progress(0)
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        st.info("视频正在推理，请等待...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            r = model(frame)
            frame_plot = r[0].plot()
            out.write(frame_plot)
            frame_idx +=1
            bar.progress(frame_idx / frame_total)
        cap.release()
        out.release()

        st.success("🎉 视频检测完成！")
        with open(out_path, "rb") as f:
            st.video(f.read())
        st.caption("带检测框的输出视频")
        st.warning("⚠️ 视频是逐帧独立计数，不会自动去重（同一条鱼跨帧会重复计数）；如果需要跟踪去重版本我再给你改！")
