import streamlit as st
import cv2
from PIL import Image
import os
import time
import csv
import pandas as pd
from ultralytics import YOLO
import tempfile

# 页面配置
st.set_page_config(
    page_title="农作物叶片病害检测系统",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 病害信息库
disease_info = {
    "健康叶片": "正常养护，无需用药",
    "白粉病": "通风降湿，喷施醚菌酯",
    "黑斑病": "减少叶面洒水，稀释多菌灵喷洒",
    "蚜虫虫害": "黄板诱杀+吡虫啉药剂",
    "缺氮黄叶": "追加氮肥、腐熟有机肥"
}

# 自定义CSS美化界面
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #2d7d46, #3a9d5a);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        color: #d4edda;
        font-size: 1.1rem;
        margin: 0;
    }
    .result-card {
        background: #f8faf8;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e8e0;
        margin-bottom: 1rem;
    }
    .disease-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.2rem;
    }
    .tag-health {
        background: #d4edda;
        color: #155724;
    }
    .tag-disease {
        background: #f8d7da;
        color: #721c24;
    }
    .tag-warning {
        background: #fff3cd;
        color: #856404;
    }
    .stButton button {
        width: 100%;
        background: #2d7d46 !important;
        color: white !important;
        font-weight: bold !important;
    }
    .stButton button:hover {
        background: #3a9d5a !important;
    }
    </style>
""", unsafe_allow_html=True)

# 标题
st.markdown("""
    <div class="main-header">
        <h1>🌿 农作物叶片病害智能检测系统</h1>
        <p>AI驱动 · 快速识别 · 精准诊断</p>
    </div>
""", unsafe_allow_html=True)

# 初始化session_state
if 'detect_history' not in st.session_state:
    st.session_state.detect_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# 侧边栏 - 功能控制
with st.sidebar:
    st.markdown("## 📋 功能控制台")

    # 模型加载
    st.markdown("### 🤖 模型加载")
    uploaded_model = st.file_uploader("上传模型文件 (best.pt)", type=['pt'])

    if uploaded_model is not None:
        # 保存上传的模型到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_file.write(uploaded_model.getvalue())
            model_path = tmp_file.name
        try:
            model = YOLO(model_path)
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.success("✅ 模型加载成功！")
        except Exception as e:
            st.error(f"❌ 模型加载失败：{str(e)}")
    else:
        # 尝试加载默认模型
        if not st.session_state.model_loaded:
            try:
                # 如果有best.pt文件，自动加载
                if os.path.exists("best.pt"):
                    model = YOLO("best.pt")
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("✅ 已自动加载模型 best.pt")
                else:
                    st.warning("⚠️ 请上传 best.pt 模型文件")
            except Exception as e:
                st.error(f"❌ 加载默认模型失败：{str(e)}")

    st.divider()

    # 参数设置
    st.markdown("### ⚙️ 参数设置")
    conf_threshold = st.slider(
        "置信度阈值",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05,
        help="调节检测灵敏度，值越高检测越严格"
    )

    st.divider()

    # 拓展工具
    st.markdown("### 🔧 拓展工具")
    if st.button("📖 病害知识库"):
        st.session_state.show_knowledge = True

    if st.button("📋 历史记录"):
        st.session_state.show_history = True

    if st.button("🗑️ 清空记录"):
        st.session_state.detect_history = []
        st.success("✅ 记录已清空")

# 主区域 - 两列布局
col1, col2 = st.columns([2, 1])

# 左侧：图像预览和检测
with col1:
    st.markdown("### 🖼️ 图像预览")

    # 图片上传
    uploaded_file = st.file_uploader(
        "选择图片",
        type=['jpg', 'png', 'jpeg', 'bmp'],
        help="支持 JPG、PNG、BMP 格式"
    )

    # 摄像头捕获
    use_camera = st.checkbox("使用摄像头")

    if use_camera:
        camera_image = st.camera_input("点击拍照")
        if camera_image is not None:
            uploaded_file = camera_image

    if uploaded_file is not None:
        # 读取图片
        image = Image.open(uploaded_file)
        st.image(image, caption="原始图片", use_container_width=True)

        # 检测按钮
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            detect_btn = st.button("🔍 AI病害检测", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🧹 清空预览", use_container_width=True)

        if clear_btn:
            st.rerun()

        if detect_btn:
            if not st.session_state.model_loaded:
                st.error("⚠️ 请先加载模型！")
            else:
                with st.spinner("🧠 模型正在分析图片..."):
                    start_time = time.time()

                    # 保存临时图片
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        image.save(tmp_file.name)
                        img_path = tmp_file.name

                    # 执行检测
                    model = st.session_state.model
                    results = model(img_path, conf=conf_threshold)[0]

                    # 计算耗时
                    process_time = round(time.time() - start_time, 2)

                    # 绘制检测结果
                    draw_frame = results.plot()
                    draw_rgb = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
                    result_image = Image.fromarray(draw_rgb)

                    # 显示检测结果
                    st.image(result_image, caption="检测结果", use_container_width=True)

                    # 显示检测信息
                    target_count = len(results.boxes)
                    st.info(f"⏱️ 检测用时：{process_time}s  |  🎯 目标数量：{target_count}")

                    # 记录结果
                    record_list = []
                    if target_count == 0:
                        st.success("✅ 叶片健康，无需用药")
                        record_list.append(["健康叶片", "100.0", disease_info["健康叶片"]])
                    else:
                        for idx, box in enumerate(results.boxes):
                            cls_id = int(box.cls[0])
                            score = round(float(box.conf[0]) * 100, 1)
                            disease_name = results.names[cls_id]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # 显示每个检测结果
                            with st.container():
                                col_a, col_b, col_c = st.columns([2, 1, 1])
                                col_a.write(f"**{disease_name}**")
                                col_b.write(f"置信度: {score}%")
                                col_c.write(f"坐标: ({x1},{y1})-({x2},{y2})")

                            record_list.append([
                                disease_name,
                                score,
                                disease_info.get(disease_name, "暂无方案")
                            ])

                    # 保存到历史记录
                    st.session_state.detect_history.append({
                        "时间": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "图片名": uploaded_file.name,
                        "检测结果": record_list
                    })

# 右侧：结果明细和知识库
with col2:
    # 结果显示
    st.markdown("### 📊 检测结果明细")

    if st.session_state.detect_history:
        # 显示最新结果
        latest = st.session_state.detect_history[-1]
        st.markdown(f"**时间：** {latest['时间']}")
        st.markdown(f"**图片：** {latest['图片名']}")

        # 表格显示
        data = []
        for item in latest['检测结果']:
            data.append({
                "病害类型": item[0],
                "置信度": f"{item[1]}%",
                "防治方案": item[2]
            })

        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            # 显示统计
            total = len(data)
            healthy = sum(1 for d in data if d["病害类型"] == "健康叶片")
            disease = total - healthy
            st.metric("总检测目标", total)
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("健康叶片", healthy)
            col_m2.metric("病害目标", disease, delta_color="inverse")
    else:
        st.info("暂无检测结果，请上传图片进行检测")

    st.divider()

    # 病害知识库
    st.markdown("### 📖 病害知识库")
    with st.expander("点击展开查看所有病害", expanded=True):
        for disease, treatment in disease_info.items():
            if disease == "健康叶片":
                st.markdown(f"✅ **{disease}**")
                st.caption(f"💊 {treatment}")
            else:
                st.markdown(f"⚠️ **{disease}**")
                st.caption(f"💊 {treatment}")
            st.divider()

# 底部 - 历史记录（展开显示）
if st.session_state.get('show_history', False):
    st.divider()
    st.markdown("### 📋 历史检测记录")
    if st.session_state.detect_history:
        # 展开所有历史记录
        for i, record in enumerate(reversed(st.session_state.detect_history[-20:])):
            with st.expander(f"记录 {i + 1} - {record['时间']}"):
                st.write(f"**图片：** {record['图片名']}")
                for item in record['检测结果']:
                    st.write(f"- {item[0]} (置信度: {item[1]}%)")
    else:
        st.info("暂无历史记录")

    if st.button("关闭历史记录"):
        st.session_state.show_history = False
        st.rerun()

# 知识库弹窗
if st.session_state.get('show_knowledge', False):
    st.divider()
    st.markdown("### 📖 完整病害知识库")
    for disease, treatment in disease_info.items():
        st.markdown(f"**{disease}**")
        st.write(f"💊 {treatment}")
        st.divider()

    if st.button("关闭知识库"):
        st.session_state.show_knowledge = False
        st.rerun()