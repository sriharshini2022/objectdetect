import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np
import tempfile
import time

st.set_page_config(page_title="YOLOv5 Object Detection", layout="wide")

# ================================
# 🎨 Custom Styling
# ================================
st.markdown("""
<style>
    .main-header {text-align:center; font-size:2.2rem; color:#3B82F6;}
    .stButton>button {background-color:#3B82F6; color:white; border-radius:8px;}
    .stButton>button:hover {background-color:#2563EB;}
    .info-box {background-color:#F8FAFC; padding:15px; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🔍 YOLOv5 Object Detection App</h1>', unsafe_allow_html=True)

# ================================
# ⚙️ Sidebar Configuration
# ================================
st.sidebar.header("⚙️ Detection Settings")

model_choice = st.sidebar.selectbox(
    "Select YOLOv5 Model", 
    ["yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]
)
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

source = st.sidebar.radio("Detection Source", ["Upload Image", "Upload Video", "Use Webcam"])

# ================================
# 🚀 Load Model
# ================================
@st.cache_resource
def load_model(model_name):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=False)
    model.conf = confidence
    return model

model = load_model(model_choice)
st.sidebar.success("✅ YOLOv5 model loaded successfully!")

# ================================
# 🖼️ IMAGE INPUT
# ================================
if source == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        results = model(img, size=640)
        results.render()
        st.image(results.ims[0], caption="Detected Objects", use_container_width=True)
        st.success("✅ Detection complete!")

# ================================
# 🎞️ VIDEO INPUT
# ================================
elif source == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            results.render()
            annotated_frame = results.ims[0]
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
        st.success("✅ Video processing finished!")

# ================================
# 🎥 WEBCAM STREAM
# ================================
else:
    st.sidebar.warning("Use Start/Stop buttons to control webcam stream")

    # Initialize session state
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    # Sidebar buttons
    start_button = st.sidebar.button("▶️ Start Webcam", key="start_webcam")
    stop_button = st.sidebar.button("⏹️ Stop Webcam", key="stop_webcam")

    if start_button:
        st.session_state.webcam_running = True
    if stop_button:
        st.session_state.webcam_running = False

    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    # Webcam loop
    while st.session_state.webcam_running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture webcam feed.")
            break

        results = model(frame)
        results.render()
        annotated_frame = results.ims[0]
        stframe.image(annotated_frame, channels="BGR", use_container_width=True)
        time.sleep(0.02)

    cap.release()

    if not st.session_state.webcam_running:
        st.success("✅ Webcam stopped.")
