import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os

# 1. 页面基础配置 (必须在第一行)
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
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
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
    <h1>🐟 YOLOv11 水产鱼类智能分析系统</h1>
    <p>集成 ByteTrack 目标追踪算法 · 实现动态跨帧去重计数与轨迹追踪</p>
</div>
""", unsafe_allow_html=True)

# 4. 模型加载
@st.cache_resource(show_spinner="正在加载 YOLOv11 模型引擎，请稍候...")
def load_model():
    return YOLO("best.pt")

model = load_model()

# 5. 侧边栏设计
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2983/2983804.png", width=80)
    st.markdown("### 🎛️ 控制面板")
    st.divider()
    mode = st.radio(
        "请选择分析模式：",
        ["📷 单张图像检测", "🎬 视频流追踪计数"],
        index=0
    )
    st.divider()
    st.markdown("### 📖 系统说明")
    st.info("""
    - **图像模式**：上传高清图片，系统将标记并统计鱼的数量。
    - **视频模式**：引入 **ByteTrack 算法**，为每条鱼分配唯一 ID，解决跨帧重复计数问题。
    """)

# 6. 主体功能区开发
if mode == "📷 单张图像检测":
    st.markdown("### 📸 图像上传与分析")
    up_img = st.file_uploader("请拖拽或选择鱼类图片 (JPG / PNG)", type=["jpg", "jpeg", "png"])
    
    if up_img is not None:
        img = Image.open(up_img)
        with st.spinner("🧠 AI 正在极速分析画面..."):
            res = model(img)
            count = len(res[0].boxes)
            
            st.metric(label="📊 本次识别鱼类总数", value=f"{count} 条", delta="检测完成", delta_color="normal")
            
            plot_img = res[0].plot()
            plot_img = Image.fromarray(cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB))
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📥 原始图像")
                st.image(img, use_column_width=True)
            with col2:
                st.markdown("#### 🎯 检测结果")
                st.image(plot_img, use_column_width=True)
            st.success("✅ 图像分析已完成，检测框渲染成功。")

elif mode == "🎬 视频流追踪计数":
    st.markdown("### 📽️ 视频追踪与去重计数")
    up_vid = st.file_uploader("请上传待检测短视频 (MP4格式，建议小于20MB)", type=["mp4"])
    
    if up_vid is not None:
        status_container = st.container()
        
        with status_container:
            st.info("🔄 正在初始化 ByteTrack 追踪引擎...")
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(up_vid.read())
            
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*'h264') 
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            progress_text = "🚀 AI 正在逐帧追踪..."
            bar = st.progress(0, text=progress_text)
            
            # 用于UI显示的占位符，实时更新当前识别到的去重数量
            metric_placeholder = st.empty()
            
            frame_idx = 0
            unique_fish_ids = set() # 核心：用于存放出现过的追踪ID
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 使用 track 方法替代原有的推理方法，并指定 tracker="bytetrack.yaml"
                results = model.track(frame, persist=True, tracker="bytetrack.yaml")
                
                # 提取这一帧中所有的 track_id
                if results[0].boxes.id is not None:
                    # 获取当前帧的追踪ID列表并转换为 Python list
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    # 放入集合自动去重
                    unique_fish_ids.update(track_ids)
                    
                    # 预留提取坐标和轨迹的位置 (供后续数据导出使用)
                    # boxes = results[0].boxes.xyxy.cpu().tolist()
                    # for box, track_id in zip(boxes, track_ids):
                    #     x1, y1, x2, y2 = box
                    #     center_x = (x1 + x2) / 2
                    #     center_y = (y1 + y2) / 2
                        
                frame_plot = results[0].plot()
                out.write(frame_plot)
                frame_idx += 1
                
                # 更新进度条和实时计数器
                bar.progress(frame_idx / frame_total, text=f"追踪进度: {frame_idx}/{frame_total} 帧")
                metric_placeholder.metric(label="🐟 视频累计去重鱼类数量", value=f"{len(unique_fish_ids)} 条")
                
            cap.release()
            out.release()
            
            bar.empty()
            st.success(f"🎉 视频处理完毕！总计提取到 {len(unique_fish_ids)} 个唯一的鱼类追踪 ID。")
        
        # 展示视频结果
        st.markdown("#### 🎬 最终分析视频 (带追踪 ID)")
        with open(out_path, "rb") as f:
            st.video(f.read())
