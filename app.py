import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import io
import numpy as np
import pandas as pd
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
    .stApp { background-color: #f4f9f9; font-family: 'Segoe UI', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #005c97 0%, #363795 100%);
        padding: 30px; border-radius: 15px; color: white; text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1); margin-bottom: 30px;
    }
    .main-header h1 { color: white; font-size: 2.5rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .main-header p { font-size: 1.1rem; opacity: 0.9; }
    [data-testid="stFileUploadDropzone"] { border: 2px dashed #005c97 !important; border-radius: 15px; background-color: #ffffff; transition: all 0.3s ease; }
    [data-testid="stFileUploadDropzone"]:hover { background-color: #e6f2f5; border-color: #363795 !important; }
    .element-container img { border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# 3. 渲染顶部 Banner
st.markdown("""
<div class="main-header">
    <h1>🐟 水产鱼类智能分析计数工作站</h1>
    <p>单图检测 · 批量图片统计 · 视频微小目标追踪</p>
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
        ["📷 单张图像检测", "📂 批量图片检测", "🎬 视频 SAHI 切片追踪"],
        index=1
    )
    
    st.divider()
    st.markdown("### ⚙️ 核心检测参数")
    conf_threshold = st.slider(
        "置信度阈值 (Confidence)", 
        min_value=0.01, max_value=1.0, value=0.50, step=0.05,
        help="调高该值可以过滤掉不确定的检测框，减少误报；调低可以检测出更多目标，但容易误判。"
    )

    if mode == "🎬 视频 SAHI 切片追踪":
        st.markdown("### ⚙️ SAHI 切片参数")
        slice_width = st.slider("切片宽度 (Slice Width)", 256, 1024, 512, step=128)
        slice_height = st.slider("切片高度 (Slice Height)", 256, 1024, 512, step=128)
        overlap = st.slider("重叠率 (Overlap Ratio)", 0.1, 0.5, 0.2, step=0.05)

# --- SAHI 推理回调函数 ---
def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice, conf=conf_threshold, verbose=False)[0]
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
        # 修复：use_column_width 改为 use_container_width
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
            
            st.success("✅ 批量检测全部完成！")
            st.metric(label="📊 本批次累计识别总鱼数", value=f"{total_fish_count} 条", delta=f"共分析 {len(uploaded_files)} 张图片")
            
            df = pd.DataFrame(results_data)
            st.markdown("#### 📑 批量统计报表")
            st.dataframe(df, use_container_width=True)
            
            # Excel 格式下载
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

elif mode == "🎬 视频 SAHI 切片追踪":
    st.markdown("### 📽️ 视频 SAHI 切片推理与追踪")
    st.caption("💡 提示：由于云端资源限制（免费版仅1核CPU），处理视频可能较慢或触发限流。建议上传 10 秒以内、分辨率不要过大的测试视频。")
    
    up_video = st.file_uploader("请上传视频文件", type=["mp4", "avi", "mov", "mkv"])
    
    if up_video is not None:
        # 将上传的视频保存为临时文件，供 cv2 读取
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(up_video.read())
        video_path = tfile.name
        
        # 读取视频基本信息
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 原视频展示
        st.video(up_video)
        
        if st.button("🚀 开始切片追踪", type="primary"):
            st.info("正在初始化 SAHI 切片器与追踪器...")
            
            # 初始化 SAHI 切片器 和 ByteTrack 追踪器
            overlap_w = int(slice_width * overlap)
            overlap_h = int(slice_height * overlap)
            
            slicer = sv.InferenceSlicer(
                callback=callback,
                slice_wh=(slice_width, slice_height),
                overlap_wh=(overlap_w, overlap_h)
            )
            tracker = sv.ByteTrack()
            box_annotator = sv.BoxAnnotator(thickness=2)
            
            # 准备输出视频（保存为临时文件）
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            
            progress_bar = st.progress(0, text="视频处理中...")
            frame_placeholder = st.empty()
            
            frame_count = 0
            total_fish_count = 0
            
            # 逐帧处理视频
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 将 BGR 转为 RGB 供 SAHI 处理
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 使用 SAHI 进行切片推理
                detections = slicer(frame_rgb)
                
                # 使用 ByteTrack 更新追踪状态
                detections = tracker.update_with_detections(detections)
                total_fish_count += len(detections)
                
                # 在帧上绘制框和追踪 ID
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # 防御性标签生成：防止 tracker_id 为 None 时引发异常
                labels = []
                if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                    for class_name, tracker_id in zip(detections.data["class_name"], detections.tracker_id):
                        if tracker_id is not None:
                            labels.append(f"{class_name} #{tracker_id}")
                        else:
                            labels.append(class_name)
                else:
                    labels = detections.data.get("class_name", ["fish"] * len(detections))
                
                annotated_frame = box_annotator.annotate(
                    scene=frame_bgr,
                    detections=detections,
                    labels=labels
                )
                
                # 写入输出视频
                out.write(annotated_frame)
                
                frame_count += 1
                if frame_count % 5 == 0 or frame_count == total_frames:
                    progress_bar.progress(min(frame_count / total_frames, 1.0), text=f"正在处理: {frame_count}/{total_frames} 帧")
                    frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="实时处理预览", width=600)
            
            # 释放资源
            cap.release()
            out.release()
            os.unlink(video_path)
            
            progress_bar.progress(1.0, text="✅ 视频处理完成！")
            st.success("视频追踪处理完毕！")
            st.metric(label="📊 追踪统计（累加值）", value=f"共追踪到 {total_fish_count} 次目标出现")
            
            # 读取输出视频并提供下载
            with open(out_path, "rb") as f:
                video_bytes = f.read()
            
            st.download_button(
                label="📥 下载追踪后的视频",
                data=video_bytes,
                file_name="fish_tracked_video.mp4",
                mime="video/mp4",
                type="primary"
            )
            
            # 预览处理后的视频
            st.video(video_bytes)
            
            # 清理输出临时文件
            os.unlink(out_path)
