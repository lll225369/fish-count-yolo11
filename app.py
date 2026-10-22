import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import io
import gc
import numpy as np
import pandas as pd
import supervision as sv
from trackers import ByteTrackTracker

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
    .stApp { background-color: #f4f9f9; font-family: 'Segoe UI', sans-serif; }
    .main-header {
        display: flex;
        justify-content: space-between; 
        align-items: center;            
        background: linear-gradient(135deg, #005c97 0%, #363795 100%);
        padding: 20px 40px; 
        border-radius: 15px; 
        color: white; 
        box-shadow: 0 10px 20px rgba(0,0,0,0.1); 
        margin-bottom: 30px;
    }
    .header-text { text-align: center; flex-grow: 1; }
    .main-header h1 { color: white; font-size: 2.5rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .main-header p { font-size: 1.1rem; opacity: 0.9; margin: 0; }
    
    .logo {
        height: 90px;         
        max-width: 120px;     
        object-fit: contain;  
        margin: 0 20px;       
    }
    @media (max-width: 768px) {
        .main-header { flex-direction: column; padding: 20px; }
        .logo { height: 60px; margin: 10px 0; }
        .main-header h1 { font-size: 1.8rem; }
    }
    [data-testid="stFileUploadDropzone"] { border: 2px dashed #005c97 !important; border-radius: 15px; background-color: #ffffff; transition: all 0.3s ease; }
    [data-testid="stFileUploadDropzone"]:hover { background-color: #e6f2f5; border-color: #363795 !important; }
    .element-container img { border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# 3. 渲染顶部 Banner（带左右校徽）
# ⚠️【重要修改点】请把你准备好的 Base64 代码分别粘贴到下面两个变量中！
# =========================================================================
logo_left_url = "data:image/png;base64,在这里粘贴左校徽的Base64代码"
logo_right_url = "data:image/png;base64,在这里粘贴右校徽的Base64代码"

st.markdown(f"""
<div class="main-header">
    <img src="{logo_left_url}" class="logo" alt="左校徽">
    <div class="header-text">
        <h1>🐟 水产鱼类智能分析计数工作站</h1>
        <p>单图检测 · 批量图片统计 · 视频微小目标追踪</p>
    </div>
    <img src="{logo_right_url}" class="logo" alt="右校徽">
</div>
""", unsafe_allow_html=True)

# 4. 模型加载
@st.cache_resource(show_spinner="正在加载 YOLOv11 模型引擎，请稍候...")
def load_model():
    return YOLO("best.pt")

model = load_model()

# 5. 侧边栏设计
with st.sidebar:
    st.markdown("### 🎛️ 控制面板")
    st.divider()
    
    mode = st.radio(
        "请选择分析模式：",
        ["📷 单张图像检测", "📂 批量图片检测", "🎬 视频智能追踪"],
        index=1
    )
    
    st.divider()
    st.markdown("### ⚙️ 核心检测参数")
    
    # --- 置信度：滑块 + 输入框 ---
    st.markdown("**置信度阈值 (Confidence)**")
    col1, col2 = st.columns([3, 1])
    with col1:
        conf_slider = st.slider("Conf", min_value=0.01, max_value=1.0, value=0.50, step=0.05, label_visibility="collapsed")
    with col2:
        conf_threshold = st.number_input("ConfNum", min_value=0.01, max_value=1.0, value=conf_slider, step=0.05, format="%.2f", label_visibility="collapsed")

    # 视频模式特有的设置
    if mode == "🎬 视频智能追踪":
        st.markdown("### ⚙️ 视频推理模式")
        video_mode = st.radio(
            "请选择目标特征：",
            ["大目标（常规整图）", "小目标（SAHI 切片）"],
            help="大目标直接整图推理，速度快；密集小目标使用切片推理，防止漏检。"
        )
        
        # 只有选了小目标，才显示切片参数
        if video_mode == "小目标（SAHI 切片）":
            st.markdown("---")
            
            # --- 切片宽度：滑块 + 输入框 ---
            st.markdown("**切片宽度 (Slice Width)**")
            c1, c2 = st.columns([3, 1])
            with c1:
                slice_w_slider = st.slider("W", min_value=256, max_value=1024, value=512, step=128, label_visibility="collapsed")
            with c2:
                slice_width = st.number_input("WNum", min_value=256, max_value=1024, value=slice_w_slider, step=128, label_visibility="collapsed")
            
            # --- 切片高度：滑块 + 输入框 ---
            st.markdown("**切片高度 (Slice Height)**")
            c1, c2 = st.columns([3, 1])
            with c1:
                slice_h_slider = st.slider("H", min_value=256, max_value=1024, value=512, step=128, label_visibility="collapsed")
            with c2:
                slice_height = st.number_input("HNum", min_value=256, max_value=1024, value=slice_h_slider, step=128, label_visibility="collapsed")

            # --- 重叠率：滑块 + 输入框 ---
            st.markdown("**重叠率 (Overlap Ratio)**")
            c1, c2 = st.columns([3, 1])
            with c1:
                overlap_slider = st.slider("O", min_value=0.1, max_value=0.5, value=0.2, step=0.05, label_visibility="collapsed")
            with c2:
                overlap = st.number_input("ONum", min_value=0.1, max_value=0.5, value=overlap_slider, step=0.05, format="%.2f", label_visibility="collapsed")

# --- 提取通用预测逻辑（支持切片或整图） ---
def predict_frame(frame_rgb, use_sahi=False):
    if use_sahi:
        overlap_w = int(slice_width * overlap)
        overlap_h = int(slice_height * overlap)
        slicer = sv.InferenceSlicer(
            callback=lambda img: sv.Detections.from_ultralytics(model(img, conf=conf_threshold, verbose=False)[0]),
            slice_wh=(slice_width, slice_height),
            overlap_wh=(overlap_w, overlap_h)
        )
        return slicer(frame_rgb)
    else:
        result = model(frame_rgb, conf=conf_threshold, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

# 6. 主体功能区开发
if mode == "📷 单张图像检测":
    st.markdown("### 📸 单图识别与分析")
    up_img = st.file_uploader("请拖拽或选择单张图片", type=["jpg", "jpeg", "png"])
    if up_img:
        img = Image.open(up_img)
        res = model(img, conf=conf_threshold)
        count = len(res[0].boxes)
        st.metric(label="📊 本次识别数量", value=f"{count} 条")
        plot_img = Image.fromarray(cv2.cvtColor(res[0].plot(), cv2.COLOR_BGR2RGB))
        col1, col2 = st.columns(2)
        with col1: st.image(img, caption="原始图片", use_container_width=True)
        with col2: st.image(plot_img, caption="检测结果", use_container_width=True)

elif mode == "📂 批量图片检测":
    st.markdown("### 📂 文件夹内多图批量识别与统计")
    st.caption("💡 操作提示：点击下方上传区域，按 `Ctrl+A` 全选文件夹内的所有图片。")
    
    uploaded_files = st.file_uploader("请批量选择/拖拽文件夹内的图片", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.info(f"📁 成功加载 {len(uploaded_files)} 张图片，等待执行批量分析。")
        
        if st.button("🚀 开始批量检测", type="primary"):
            progress_bar = st.progress(0, text="准备开始处理...")
            results_data = [] 
            preview_placeholder = st.empty() 
            total_fish_count = 0
            
            for i, file in enumerate(uploaded_files):
                img = Image.open(file)
                res = model(img, conf=conf_threshold)
                count = len(res[0].boxes)
                total_fish_count += count
                
                results_data.append({
                    "文件名": file.name,
                    "识别数量 (条)": count,
                    "过滤阈值": f"{conf_threshold:.2f}"
                })
                
                progress_bar.progress((i + 1) / len(uploaded_files), text=f"正在处理: {file.name} ({i+1}/{len(uploaded_files)})")
                
                plot_img = Image.fromarray(cv2.cvtColor(res[0].plot(), cv2.COLOR_BGR2RGB))
                preview_placeholder.image(plot_img, caption=f"最新处理预览: {file.name} - {count} 条", width=600)
                
                del img, res, plot_img
                gc.collect()
            
            st.success("✅ 批量检测全部完成！")
            st.metric(label="📊 本批次累计识别总鱼数", value=f"{total_fish_count} 条", delta=f"共分析 {len(uploaded_files)} 张图片")
            
            df = pd.DataFrame(results_data)
            st.markdown("#### 📑 批量统计报表")
            st.dataframe(df, use_container_width=True)
            
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='检测统计结果')
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.download_button(
                    label="📊 下载 Excel 统计报表",
                    data=excel_buffer.getvalue(),
                    file_name="fish_batch_detection_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )

elif mode == "🎬 视频智能追踪":
    st.markdown("### 📽️ 视频智能追踪")
    st.caption(f"💡 当前推理模式：**{video_mode}** | 置信度：**{conf_threshold:.2f}**")
    st.caption("⚠️ 提示：云端资源受限，建议上传 10 秒以内、分辨率不过大的视频。")
    
    up_video = st.file_uploader("请上传视频文件", type=["mp4", "avi", "mov", "mkv"])
    
    if up_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(up_video.read())
        video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        MAX_FRAMES = 300 # 限制处理帧数
        process_frames = min(total_frames, MAX_FRAMES)
        
        st.video(up_video)
        
        if st.button("🚀 开始视频追踪", type="primary"):
            st.info(f"正在初始化追踪器（{video_mode}）...")
            
            use_sahi = True if video_mode == "小目标（SAHI 切片）" else False
            
            tracker = ByteTrackTracker()
            box_annotator = sv.BoxAnnotator(thickness=2)
            label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=2)
            
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            
            progress_bar = st.progress(0, text="视频处理中...")
            frame_placeholder = st.empty()
            
            frame_count = 0
            total_fish_count = 0
            
            try:
                while cap.isOpened() and frame_count < process_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detections = predict_frame(frame_rgb, use_sahi=use_sahi)
                    
                    detections = tracker.update(detections)
                    total_fish_count += len(detections)
                    
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    
                    labels = []
                    if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
                            class_name = model.names[int(class_id)] if hasattr(model, 'names') else "fish"
                            if tracker_id is not None:
                                labels.append(f"{class_name} #{tracker_id}")
                            else:
                                labels.append(class_name)
                    else:
                        if hasattr(detections, 'class_id') and detections.class_id is not None:
                            labels = [model.names[int(cid)] if hasattr(model, 'names') else "fish" for cid in detections.class_id]
                        else:
                            labels = ["fish"] * len(detections)
                    
                    annotated_frame = box_annotator.annotate(scene=frame_bgr, detections=detections)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                    
                    out.write(annotated_frame)
                    
                    frame_count += 1
                    if frame_count % 5 == 0 or frame_count == process_frames:
                        progress_bar.progress(min(frame_count / process_frames, 1.0), text=f"正在处理: {frame_count}/{process_frames} 帧")
                        frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="实时处理预览", width=600)
                    
                    del frame, frame_rgb, frame_bgr, detections, annotated_frame
                    gc.collect()

            finally:
                cap.release()
                out.release()
                try:
                    os.unlink(video_path)
                except:
                    pass
            
            progress_bar.progress(1.0, text="✅ 视频处理完成！")
            st.success("视频追踪处理完毕！")
            st.metric(label="📊 追踪统计（累加值）", value=f"共追踪到 {total_fish_count} 次目标出现")
            
            with open(out_path, "rb") as f:
                video_bytes = f.read()
            
            st.download_button(
                label="📥 下载追踪后的视频",
                data=video_bytes,
                file_name="fish_tracked_video.mp4",
                mime="video/mp4",
                type="primary"
            )
            
            st.video(video_bytes)
            
            try:
                os.unlink(out_path)
            except:
                pass
