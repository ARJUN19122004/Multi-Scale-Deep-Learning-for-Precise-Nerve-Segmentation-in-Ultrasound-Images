import streamlit as st
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import time
import base64
import os

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Nerve Segmentation System",
    page_icon="🧠🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# Background Video
# -------------------------
def get_base64_video(video_path):
    with open(video_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_video_path = os.path.join(os.path.dirname(__file__), "IMG_9619.mp4")
bg_video_base64 = get_base64_video(bg_video_path)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

.stApp {{
    background: transparent !important;
    font-family: 'Inter', sans-serif;
}}

#bg-video {{
    position: fixed;
    right: 0;
    bottom: 0;
    min-width: 100%;
    min-height: 100%;
    object-fit: cover;
    z-index: -1;
}}

/* Hide sidebar */
section[data-testid="stSidebar"] {{ display: none !important; }}
button[data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent !important; }}
footer {{ display: none !important; }}

/* Hero */
.hero-container {{
    text-align: center;
    padding: 3rem 2rem;
    background: rgba(0,0,0,0.6);
    border-radius: 24px;
    backdrop-filter: blur(15px);
    margin-bottom: 2rem;
}}
.hero-title {{
    font-size: 3rem;
    font-weight: 800;
    color: white;
}}
.hero-subtitle {{
    font-size: 1.1rem;
    color: #e2e8f0;
}}

/* Metric Cards */
.stats-row {{
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin: 2rem 0;
    flex-wrap: wrap;
}}

.stat-card {{
    background: rgba(0, 0, 0, 0.65);
    border-radius: 18px;
    padding: 1.5rem 2rem;
    text-align: center;
    min-width: 180px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 35px rgba(0,0,0,0.4);
}}

.stat-value {{
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff;
    margin-top: 0.5rem;
}}

.stat-label {{
    font-size: 0.8rem;
    color: #cbd5e1;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* Result Cards */
.result-card {{
    background: rgba(0,0,0,0.65);
    border-radius: 20px;
    padding: 1.2rem;
    backdrop-filter: blur(12px);
}}

.result-card-title {{
    font-size: 1rem;
    font-weight: 600;
    color: white;
    margin-bottom: 1rem;
}}

</style>

<video autoplay loop muted playsinline id="bg-video">
    <source src="data:video/mp4;base64,{bg_video_base64}" type="video/mp4">
</video>
""", unsafe_allow_html=True)

# -------------------------
# Model Setup
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1
    ).to(DEVICE)

    model.load_state_dict(
        torch.load("nerve_model.pth", map_location=DEVICE)
    )
    model.eval()
    return model

model = load_model()

# -------------------------
# Hero Section
# -------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Nerve Segmentation System</div>
    <div class="hero-subtitle">
        Upload an ultrasound image and detect nerve structures automatically.
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Upload
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    label_visibility="collapsed"
)

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)) / 255.0
    tensor = torch.tensor(resized).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)
        raw_mask = pred.squeeze().cpu().numpy()

    mask = (raw_mask > 0.5).astype(np.uint8)
    mask_full = cv2.resize(mask, (img.shape[1], img.shape[0]))

    green = np.zeros_like(img)
    green[:, :, 1] = 255

    overlay = np.where(
        mask_full[:, :, None] == 1,
        cv2.addWeighted(img, 0.6, green, 0.4, 0),
        img
    )

    total_pixels = mask_full.size
    nerve_pixels = int(np.sum(mask_full))
    coverage = (nerve_pixels / total_pixels) * 100
    avg_conf = float(np.mean(raw_mask[raw_mask > 0.5])) * 100 if nerve_pixels > 0 else 0

    # -------------------------
    # Glass Metric Cards
    # -------------------------
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-label">📐 Image Size</div>
            <div class="stat-value">{img.shape[1]} × {img.shape[0]}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">🧬 Nerve Pixels</div>
            <div class="stat-value">{nerve_pixels:,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">📊 Coverage</div>
            <div class="stat-value">{coverage:.2f}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">🎯 Avg Confidence</div>
            <div class="stat-value">{avg_conf:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="result-card"><div class="result-card-title">Original Image</div></div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.markdown('<div class="result-card"><div class="result-card-title">Nerve Overlay</div></div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB), use_container_width=True)
