import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import io
import gc
import zipfile
import numpy as np
import pandas as pd
import supervision as sv
from trackers import ByteTrackTracker

# 1. 页面基础配置
st.set_page_config(
    page_title="金鱼智能识别与追踪系统",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 高级自定义 CSS 美化（完美兼容日夜模式）
st.markdown("""
<style>
    .stApp { font-family: 'Segoe UI', sans-serif; }

    .main-header {
        display: flex;
        justify-content: space-between; 
        align-items: center;            
        background: linear-gradient(135deg, #005c97 0%, #363795 100%);
        padding: 20px 40px; 
        border-radius: 15px; 
        color: white !important; 
        box-shadow: 0 10px 20px rgba(0,0,0,0.1); 
        margin-bottom: 30px;
    }
    
    .main-header h1 { 
        color: white !important; 
        font-size: 2.5rem; 
        margin-bottom: 10px; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3); 
    }
    
    .main-header p { 
        color: white !important; 
        font-size: 1.1rem; 
        opacity: 0.9; 
        margin: 0; 
    }
    
    .header-text { text-align: center; flex-grow: 1; }
    
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
    
    [data-testid="stFileUploadDropzone"] { 
        border: 2px dashed #005c97 !important; 
        border-radius: 15px; 
        transition: all 0.3s ease; 
    }
    [data-testid="stFileUploadDropzone"]:hover { 
        background-color: rgba(0, 92, 151, 0.1) !important; 
        border-color: #363795 !important; 
    }
    
    .element-container img { border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# 3. 渲染顶部 Banner（带左右校徽）
# ⚠️ 请把你的 Base64 代码粘贴到下面两个三引号之间！
# =========================================================================
st.markdown(f"""
<div class="main-header">
    <!-- 左侧：内嵌目标检测准星 SVG -->
    <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.7)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin: 0 20px;">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="2" x2="12" y2="6"></line>
        <line x1="12" y1="18" x2="12" y2="22"></line>
        <line x1="2" y1="12" x2="6" y2="12"></line>
        <line x1="18" y1="12" x2="22" y2="12"></line>
        <circle cx="12" cy="12" r="3" fill="rgba(255,255,255,0.3)"></circle>
    </svg>

    <div class="header-text">
        <h1>🐟 水产鱼类智能分析计数工作站</h1>
        <p>单图检测 · 批量图片统计 · 视频微小目标追踪</p>
    </div>

    <!-- 右侧：内嵌鱼/数据 SVG -->
    <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.7)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin: 0 20px;">
        <path d="M22 12s-3-5-10-5-9 5-9 5 3 5 9 5 10-5 10-5z"></path>
        <path d="M2 12c2.5 0 4 2 5 3"></path>
        <path d="M2 12c2.5 0 4-2 5-3"></path>
        <circle cx="15" cy="12" r="1.5" fill="rgba(255,255,255,0.5)"></circle>
    </svg>
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
    
    # --- 置信度双向联动 ---
    if "conf_val" not in st.session_state:
        st.session_state.conf_val = 0.50
        st.session_state.conf_slider = 0.50
        st.session_state.conf_num = 0.50

    def sync_conf_slider():
        st.session_state.conf_val = st.session_state.conf_slider
        st.session_state.conf_num = st.session_state.conf_slider

    def sync_conf_num():
        st.session_state.conf_val = st.session_state.conf_num
        st.session_state.conf_slider = st.session_state.conf_num
        
    def reset_conf():
        st.session_state.conf_val = 0.50
        st.session_state.conf_slider = 0.50
        st.session_state.conf_num = 0.50

    st.markdown("**置信度阈值 (Confidence)**")
    col1, col2, col3 = st.columns([3, 1.5, 1])
    with col1:
        st.slider("Conf", min_value=0.01, max_value=1.0, value=st.session_state.conf_val, step=0.05, label_visibility="collapsed", key="conf_slider", on_change=sync_conf_slider)
    with col2:
        st.number_input("ConfNum", min_value=0.01, max_value=1.0, value=st.session_state.conf_val, step=0.05, format="%.2f", label_visibility="collapsed", key="conf_num", on_change=sync_conf_num)
    with col3:
        st.button("🔄", key="reset_conf_btn", on_click=reset_conf, help="恢复默认值 (0.50)")
    
    conf_threshold = st.session_state.conf_val

    # 视频模式特有的设置
    if mode == "🎬 视频智能追踪":
        st.markdown("### ⚙️ 视频推理模式")
        video_mode = st.radio(
            "请选择目标特征：",
            ["大目标（常规整图）", "小目标（SAHI 切片）"],
            help="大目标直接整图推理，速度快；密集小目标使用切片推理，防止漏检。"
        )
        
        if video_mode == "小目标（SAHI 切片）":
            st.markdown("---")
            
            # --- 切片宽度双向联动 ---
            if "slice_w_val" not in st.session_state:
                st.session_state.slice_w_val = 512
                st.session_state.w_slider = 512
                st.session_state.w_num = 512

            def sync_w_slider():
                st.session_state.slice_w_val = st.session_state.w_slider
                st.session_state.w_num = st.session_state.w_slider

            def sync_w_num():
                st.session_state.slice_w_val = st.session_state.w_num
                st.session_state.w_slider = st.session_state.w_num
                
            def reset_w():
                st.session_state.slice_w_val = 512
                st.session_state.w_slider = 512
                st.session_state.w_num = 512

            st.markdown("**切片宽度 (Slice Width)**")
            c1, c2, c3 = st.columns([3, 1.5, 1])
            with c1:
                st.slider("W", min_value=256, max_value=1024, value=st.session_state.slice_w_val, step=128, label_visibility="collapsed", key="w_slider", on_change=sync_w_slider)
            with c2:
                st.number_input("WNum", min_value=256, max_value=1024, value=st.session_state.slice_w_val, step=128, label_visibility="collapsed", key="w_num", on_change=sync_w_num)
            with c3:
                st.button("🔄", key="reset_w_btn", on_click=reset_w, help="恢复默认值 (512)")
            slice_width = st.session_state.slice_w_val

            # --- 切片高度双向联动 ---
            if "slice_h_val" not in st.session_state:
                st.session_state.slice_h_val = 512
                st.session_state.h_slider = 512
                st.session_state.h_num = 512

            def sync_h_slider():
                st.session_state.slice_h_val = st.session_state.h_slider
                st.session_state.h_num = st.session_state.h_slider

            def sync_h_num():
                st.session_state.slice_h_val = st.session_state.h_num
                st.session_state.h_slider = st.session_state.h_num
                
            def reset_h():
                st.session_state.slice_h_val = 512
                st.session_state.h_slider = 512
                st.session_state.h_num = 512

            st.markdown("**切片高度 (Slice Height)**")
            c1, c2, c3 = st.columns([3, 1.5, 1])
            with c1:
                st.slider("H", min_value=256, max_value=1024, value=st.session_state.slice_h_val, step=128, label_visibility="collapsed", key="h_slider", on_change=sync_h_slider)
            with c2:
                st.number_input("HNum", min_value=256, max_value=1024, value=st.session_state.slice_h_val, step=128, label_visibility="collapsed", key="h_num", on_change=sync_h_num)
            with c3:
                st.button("🔄", key="reset_h_btn", on_click=reset_h, help="恢复默认值 (512)")
            slice_height = st.session_state.slice_h_val

            # --- 重叠率双向联动 ---
            if "overlap_val" not in st.session_state:
                st.session_state.overlap_val = 0.20
                st.session_state.o_slider = 0.20
                st.session_state.o_num = 0.20

            def sync_o_slider():
                st.session_state.overlap_val = st.session_state.o_slider
                st.session_state.o_num = st.session_state.o_slider

            def sync_o_num():
                st.session_state.overlap_val = st.session_state.o_num
                st.session_state.o_slider = st.session_state.o_num
                
            def reset_o():
                st.session_state.overlap_val = 0.20
                st.session_state.o_slider = 0.20
                st.session_state.o_num = 0.20

            st.markdown("**重叠率 (Overlap Ratio)**")
            c1, c2, c3 = st.columns([3, 1.5, 1])
            with c1:
                st.slider("O", min_value=0.1, max_value=0.5, value=st.session_state.overlap_val, step=0.05, label_visibility="collapsed", key="o_slider", on_change=sync_o_slider)
            with c2:
                st.number_input("ONum", min_value=0.1, max_value=0.5, value=st.session_state.overlap_val, step=0.05, format="%.2f", label_visibility="collapsed", key="o_num", on_change=sync_o_num)
            with c3:
                st.button("🔄", key="reset_o_btn", on_click=reset_o, help="恢复默认值 (0.20)")
            overlap = st.session_state.overlap_val

# --- 提取通用预测逻辑 ---
def predict_frame(frame_rgb, use_sahi=False):
    if use_sahi:
        overlap_w = int(slice_width * overlap)
        overlap_h = int(slice_height * overlap)
        
        def slice_callback(img_slice):
            results = model(img_slice, conf=conf_threshold, verbose=False)
            res = results[0] if isinstance(results, list) else results
            return sv.Detections.from_ultralytics(res)
            
        slicer = sv.InferenceSlicer(
            callback=slice_callback,
            slice_wh=(slice_width, slice_height),
            overlap_wh=(overlap_w, overlap_h)
        )
        return slicer(frame_rgb)
    else:
        results = model(frame_rgb, conf=conf_threshold, verbose=False)
        res = results[0] if isinstance(results, list) else results
        return sv.Detections.from_ultralytics(res)

# 6. 主体功能区开发
if mode == "📷 单张图像检测":
    st.markdown("### 📸 单图识别与分析")
    up_img = st.file_uploader("请拖拽或选择单张图片", type=["jpg", "jpeg", "png"])
    
    if up_img:
        img = Image.open(up_img)
        
        # 如果更换了图片，清除之前的预测缓存
        if "current_img_name" not in st.session_state or st.session_state.current_img_name != up_img.name:
            st.session_state.current_img_name = up_img.name
            st.session_state.single_pred_img = None
            st.session_state.single_pred_count = 0

        col1, col2 = st.columns(2)
        with col1: 
            st.image(img, caption="原始图片", use_container_width=True)
        
        if st.button("🚀 开始/重新预测", type="primary"):
            with st.spinner("正在推理中..."):
                results = model(img, conf=conf_threshold)
                res = results[0] if isinstance(results, list) else results
                
                # 终极防错：通过 supervision 标准对象获取准确数量
                detections = sv.Detections.from_ultralytics(res)
                count = len(detections)
                
                plot_img = Image.fromarray(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB))
                
                st.session_state.single_pred_img = plot_img
                st.session_state.single_pred_count = count
        
        if st.session_state.single_pred_img is not None:
            st.metric(label="📊 本次识别数量 (当前阈值: {:.2f})".format(conf_threshold), value=f"{st.session_state.single_pred_count} 条")
            
            with col2:
                st.image(st.session_state.single_pred_img, caption="检测结果", use_container_width=True)
                
            # 【单图下载】
            img_buffer = io.BytesIO()
            st.session_state.single_pred_img.save(img_buffer, format="JPEG")
            st.download_button(
                label="📥 下载检测结果图",
                data=img_buffer.getvalue(),
                file_name=f"result_{up_img.name.rsplit('.', 1)[0]}.jpg",
                mime="image/jpeg",
                type="primary"
            )
        else:
            with col2:
                st.info("👈 请调整好侧边栏参数后，点击“开始/重新预测”按钮")

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
            
            image_results_bytes = [] 
            
            for i, file in enumerate(uploaded_files):
                img = Image.open(file)
                results = model(img, conf=conf_threshold)
                res = results[0] if isinstance(results, list) else results
                
                # 终极防错：通过 supervision 标准对象获取准确数量
                detections = sv.Detections.from_ultralytics(res)
                count = len(detections)
                total_fish_count += count
                
                results_data.append({
                    "文件名": file.name,
                    "识别数量 (条)": count,
                    "过滤阈值": f"{conf_threshold:.2f}"
                })
                
                progress_bar.progress((i + 1) / len(uploaded_files), text=f"正在处理: {file.name} ({i+1}/{len(uploaded_files)})")
                
                plot_img = Image.fromarray(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB))
                preview_placeholder.image(plot_img, caption=f"最新处理预览: {file.name} - {count} 条", width=600)
                
                img_buffer = io.BytesIO()
                plot_img.save(img_buffer, format="JPEG")
                image_results_bytes.append({
                    "filename": f"pred_{file.name.rsplit('.', 1)[0]}.jpg",
                    "bytes": img_buffer.getvalue()
                })
                
                del img, res, plot_img
                gc.collect()
            
            st.session_state["batch_results"] = image_results_bytes
            st.session_state["current_preview_index"] = 0
            
            st.success("✅ 批量检测全部完成！")
            st.metric(label="📊 本批次累计识别总鱼数", value=f"{total_fish_count} 条", delta=f"共分析 {len(uploaded_files)} 张图片")
            
            # 【表格输出】
            df = pd.DataFrame(results_data)
            st.markdown("#### 📑 批量统计报表")
            st.dataframe(df, use_container_width=True)
            
            # ================= 导出功能模块 =================
            st.markdown("#### 📥 结果导出")
            col_excel, col_zip = st.columns(2)
            
            # 【Excel 下载】
            with col_excel:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='检测统计结果')
                st.download_button(
                    label="📊 下载 Excel 统计报表",
                    data=excel_buffer.getvalue(),
                    file_name="fish_batch_detection_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # 【ZIP 打包下载】
            with col_zip:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for item in image_results_bytes:
                        zip_file.writestr(item["filename"], item["bytes"])
                
                st.download_button(
                    label="📦 下载所有预测结果图 (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="fish_batch_predicted_images.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True
                )

    # ================= 逐张查看与单独下载模块 =================
    if "batch_results" in st.session_state and st.session_state["batch_results"]:
        st.divider()
        st.markdown("#### 🔍 逐张查看预测结果")
        
        results = st.session_state["batch_results"]
        total_images = len(results)
        
        if st.session_state["current_preview_index"] >= total_images:
            st.session_state["current_preview_index"] = 0
            
        current_idx = st.session_state["current_preview_index"]
        
        col_btn_prev, col_select, col_btn_next = st.columns([1, 4, 1])
        
        with col_btn_prev:
            if st.button("◀ 上一张", use_container_width=True):
                st.session_state["current_preview_index"] = (current_idx - 1) % total_images
                st.rerun()
                
        with col_select:
            def on_select_change():
                st.session_state["current_preview_index"] = st.session_state["selectbox_preview"]

            st.selectbox(
                "选择图片：", 
                options=range(total_images),
                format_func=lambda x: f"[{x+1}/{total_images}] {results[x]['filename']}",
                index=current_idx,
                key="selectbox_preview",
                on_change=on_select_change,
                label_visibility="collapsed"
            )
            
        with col_btn_next:
            if st.button("下一张 ▶", use_container_width=True):
                st.session_state["current_preview_index"] = (current_idx + 1) % total_images
                st.rerun()
                
        current_item = results[st.session_state["current_preview_index"]]
        
        # 显示图片
        st.image(
            current_item["bytes"], 
            caption=f"当前查看: {current_item['filename']}",
            use_container_width=True
        )
        
        # 【单张图片单独下载】
        st.download_button(
            label=f"📥 下载当前单张结果图 ({current_item['filename']})",
            data=current_item["bytes"],
            file_name=current_item["filename"],
            mime="image/jpeg"
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
        
        MAX_FRAMES = 300
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
