import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import numpy as np
import pandas as pd  # 新增：用于处理批量结果表格和导出
import supervision as sv

# 1. 页面基础配置
st.set_page_config(
    page_title="鱼类智能识别与追踪系统",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 高级自定义 CSS 美化
st.markdown("""
<style>
    .stApp {
        background-color: #f4f9f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #005c97 0%, #363795 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #005c97 !important;
        border-radius: 15px;
        background-color: #ffffff;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: #e6f2f5;
        border-color: #363795 !important;
    }
    .element-container img {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 3. 渲染精致的顶部 Banner
st.markdown("""
<div class="main-header">
    <h1>🐟 水产鱼类智能分析全能工作站</h1>
    <p>单图检测 · 批量图片统计 · 视频微小目标追踪</p>
</div>
""", unsafe_allow_html=True)

# 4. 模型加载
@st.cache_resource(show_spinner="正在加载 YOLOv11 模型引擎，请稍候...")
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- SAHI 推理回调函数 ---
def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice, verbose=False)[0]
    return sv.Detections.from_ultralytics(result)

# 5. 侧边栏设计
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2983/2983804.png", width=80)
    st.markdown("### 🎛️ 控制面板")
    st.divider()
    
    # 【新增】加入批量图片检测模式
    mode = st.radio(
        "请选择分析模式：",
        ["📷 单张图像检测", "📂 批量图片检测", "🎬 视频 SAHI 切片追踪"],
        index=1
    )
    
    # 动态参数调节区 (针对视频模式)
    if mode == "🎬 视频 SAHI 切片追踪":
        st.markdown("### ⚙️ SAHI 切片参数")
        slice_width = st.slider("切片宽度 (Slice Width)", 256, 1024, 512, step=128)
        slice_height = st.slider("切片高度 (Slice Height)", 256, 1024, 512, step=128)
        overlap = st.slider("重叠率 (Overlap Ratio)", 0.1, 0.5, 0.2, step=0.05)
        
    st.divider()
    st.markdown("### 📖 模式说明")
    st.info("""
    - **单张图像**：精准识别单张图。
    - **批量图片**：支持拖拽整个文件夹的图片，自动生成统计报表。
    - **视频追踪**：结合 SAHI 与 ByteTrack 追踪微小鱼群。
    """)

# 6. 主体功能区开发

# ================= 模式 1: 单张图像 =================
if mode == "📷 单张图像检测":
    st.markdown("### 📸 单图识别与分析")
    up_img = st.file_uploader("请拖拽或选择单张图片", type=["jpg", "jpeg", "png"])
    if up_img:
        img = Image.open(up_img)
        res = model(img)
        count = len(res[0].boxes)
        st.metric(label="📊 本次识别数量", value=f"{count} 条")
        plot_img = Image.fromarray(cv2.cvtColor(res[0].plot(), cv2.COLOR_BGR2RGB))
        col1, col2 = st.columns(2)
        with col1: st.image(img, caption="原始图片", use_column_width=True)
        with col2: st.image(plot_img, caption="检测结果", use_column_width=True)

# ================= 模式 2: 批量图片 (本次新增) =================
elif mode == "📂 批量图片检测":
    st.markdown("### 📂 文件夹内多图批量识别与统计")
    st.caption("💡 操作提示：点击下方上传区域，然后在弹出的文件管理器中按 `Ctrl+A` 全选文件夹内的所有图片即可。")
    
    # 允许多文件上传
    uploaded_files = st.file_uploader("请批量选择/拖拽文件夹内的图片", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.info(f"📁 成功加载 {len(uploaded_files)} 张图片，等待执行批量分析。")
        
        if st.button("🚀 开始批量检测", type="primary"):
            progress_bar = st.progress(0, text="准备开始处理...")
            results_data = []  # 存储统计结果的列表
            
            # 使用一个占位符展示当前处理的图片，避免渲染太多图导致浏览器崩溃
            preview_placeholder = st.empty() 
            
            total_fish_count = 0
            
            for i, file in enumerate(uploaded_files):
                # 读取图像
                img = Image.open(file)
                # 执行预测
                res = model(img)
                count = len(res[0].boxes)
                total_fish_count += count
                
                # 记录数据
                results_data.append({
                    "图片文件名": file.name,
                    "检测到的鱼类数量 (条)": count
                })
                
                # 更新进度条
                progress_text = f"正在处理: {file.name} ({i+1}/{len(uploaded_files)})"
                progress_bar.progress((i + 1) / len(uploaded_files), text=progress_text)
                
                # 动态预览最后一张处理完的图片
                plot_img = Image.fromarray(cv2.cvtColor(res[0].plot(), cv2.COLOR_BGR2RGB))
                preview_placeholder.image(plot_img, caption=f"最新处理预览: {file.name} - {count} 条", width=600)
            
            st.success("✅ 批量检测全部完成！")
            
            # 渲染统计面板
            st.metric(label="📊 本批次累计识别总鱼数", value=f"{total_fish_count} 条", delta=f"共分析 {len(uploaded_files)} 张图片")
            
            # 将结果转换为 Pandas DataFrame 并展示
            df = pd.DataFrame(results_data)
            st.markdown("#### 📑 批量统计报表")
            st.dataframe(df, use_container_width=True)
            
            # 提供 CSV 下载功能 (使用 utf-8-sig 防止 Excel 打开中文乱码)
            csv_data = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下载统计报表 (CSV格式)",
                data=csv_data,
                file_name="batch_detection_results.csv",
                mime="text/csv"
            )

# ================= 模式 3: 视频 SAHI =================
elif mode == "🎬 视频 SAHI 切片追踪":
    st.markdown("### 📽️ 视频 SAHI 切片推理与追踪")
    up_vid = st.file_uploader("请上传待检测短视频 (MP4格式)", type=["mp4"])
    # 此处保持上一轮回答中的完整 SAHI + ByteTrack 代码即可
    if up_vid is not None:
        st.info("请参考上一步的视频逻辑 (为保证代码清晰，此处省略了内部嵌套，直接保留之前提供的视频块即可)")
