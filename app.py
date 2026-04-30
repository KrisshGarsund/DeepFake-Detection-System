"""
DeepfakeGuard — Streamlit Web Application
Multimodal deepfake detection for images, videos, and audio.
"""
import streamlit as st
import numpy as np
import os
import sys
import tempfile
import time
import json
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import IMAGE_MODEL_PATH, AUDIO_MODEL_PATH, MODELS_DIR
from detector.pipeline import DeepfakeDetector

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="DeepfakeGuard — AI Media Authenticator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #0a1628 100%); }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #0a1628 100%);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #8892b0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .result-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    .result-real {
        border-left: 4px solid #00ff88;
        box-shadow: 0 0 20px rgba(0,255,136,0.1);
    }
    
    .result-fake {
        border-left: 4px solid #ff4444;
        box-shadow: 0 0 20px rgba(255,68,68,0.1);
    }
    
    .confidence-high { color: #ff4444; }
    .confidence-medium { color: #ffaa00; }
    .confidence-low { color: #00ff88; }
    
    .metric-box {
        background: rgba(123,47,247,0.1);
        border: 1px solid rgba(123,47,247,0.3);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #7b2ff7;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .upload-zone {
        border: 2px dashed rgba(123,47,247,0.4);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        background: rgba(123,47,247,0.05);
        transition: all 0.3s ease;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7b2ff7, #00d4ff);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(123,47,247,0.3);
    }
    
    .sidebar .sidebar-content {
        background: rgba(10,10,26,0.95);
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a, #1a0a2e);
    }
    
    .audit-entry {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-left: 3px solid #7b2ff7;
    }
</style>
""", unsafe_allow_html=True)


# ─── State Management ─────────────────────────────────────────
if "detector" not in st.session_state:
    st.session_state.detector = None
if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = []
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False


def load_detector():
    """Load the detection pipeline."""
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Loading AI detection models..."):
            try:
                st.session_state.detector = DeepfakeDetector()
                st.session_state.models_loaded = True
            except Exception as e:
                st.error(f"Error loading models: {e}")
                return None
    return st.session_state.detector


def add_to_audit(result):
    """Add detection result to audit trail."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **result
    }
    st.session_state.audit_trail.insert(0, entry)


def display_result(result):
    """Display detection result with styling."""
    pred = result.get("prediction", "Unknown")
    conf = result.get("confidence", 0)
    modality = result.get("modality", "unknown")
    explanation = result.get("explanation", "")
    fake_prob = result.get("fake_probability", 0)
    
    is_fake = pred == "FAKE"
    card_class = "result-fake" if is_fake else "result-real"
    icon = "🔴" if is_fake else "🟢"
    color = "#ff4444" if is_fake else "#00ff88"
    
    st.markdown(f"""
    <div class="result-card {card_class}">
        <h2 style="color: {color}; margin: 0;">
            {icon} {pred}
        </h2>
        <p style="color: #ccd6f6; font-size: 1.2rem; margin: 0.5rem 0;">
            Confidence: <b style="color: {color}">{conf:.1%}</b>
        </p>
        <p style="color: #8892b0; margin: 0.5rem 0;">
            Modality: <b>{modality.upper()}</b>
        </p>
        <p style="color: #8892b0; margin-top: 1rem;">
            💡 {explanation}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence gauge
    st.progress(fake_prob, text=f"Fake Probability: {fake_prob:.1%}")


# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ DeepfakeGuard")
    st.markdown("---")
    
    # Model status
    st.markdown("#### Model Status")
    
    img_status = "✅ Loaded" if os.path.exists(IMAGE_MODEL_PATH) else "❌ Not trained"
    audio_status = "✅ Loaded" if os.path.exists(AUDIO_MODEL_PATH) else "❌ Not trained"
    
    st.markdown(f"**Image Detector:** {img_status}")
    st.markdown(f"**Audio Detector:** {audio_status}")
    st.markdown(f"**Video Detector:** {'✅ Ready' if os.path.exists(IMAGE_MODEL_PATH) else '❌ Needs image model'}")
    
    st.markdown("---")
    
    # Model metrics
    st.markdown("#### Performance Metrics")
    for model_name, metrics_file in [("Image", "image_metrics.json"), ("Audio", "audio_metrics.json")]:
        metrics_path = os.path.join(MODELS_DIR, metrics_file)
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            acc = metrics.get("val_accuracy", 0)
            auc = metrics.get("val_auc", 0)
            st.markdown(f"**{model_name}:** Acc={acc:.2%} | AUC={auc:.3f}")
    
    st.markdown("---")
    
    # Threshold setting
    threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05,
                          help="Score above this = FAKE")
    
    st.markdown("---")
    st.markdown("#### Audit Trail")
    if st.session_state.audit_trail:
        for entry in st.session_state.audit_trail[:10]:
            pred = entry.get("prediction", "?")
            icon = "🔴" if pred == "FAKE" else "🟢"
            conf = entry.get("confidence", 0)
            st.markdown(
                f'<div class="audit-entry">'
                f'{icon} <b>{entry.get("file", "?")}</b><br>'
                f'{entry.get("modality", "?")} | {pred} ({conf:.1%})<br>'
                f'<small>{entry.get("timestamp", "")}</small>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown("*No analyses yet*")


# ─── Main Content ─────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🛡️ DeepfakeGuard</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">AI-Powered Multimodal Deepfake Detection — '
    'Images • Videos • Audio</p>',
    unsafe_allow_html=True
)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        '<div class="metric-box"><div class="metric-value">3</div>'
        '<div class="metric-label">Modalities</div></div>',
        unsafe_allow_html=True
    )
with col2:
    n_analyzed = len(st.session_state.audit_trail)
    st.markdown(
        f'<div class="metric-box"><div class="metric-value">{n_analyzed}</div>'
        f'<div class="metric-label">Files Analyzed</div></div>',
        unsafe_allow_html=True
    )
with col3:
    n_fake = sum(1 for e in st.session_state.audit_trail if e.get("prediction") == "FAKE")
    st.markdown(
        f'<div class="metric-box"><div class="metric-value">{n_fake}</div>'
        f'<div class="metric-label">Deepfakes Found</div></div>',
        unsafe_allow_html=True
    )
with col4:
    n_real = sum(1 for e in st.session_state.audit_trail if e.get("prediction") == "REAL")
    st.markdown(
        f'<div class="metric-box"><div class="metric-value">{n_real}</div>'
        f'<div class="metric-label">Authentic Media</div></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ─── Upload and Analysis ──────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "🎬 Video Detection", "🎤 Audio Detection"])

with tab1:
    st.markdown("### Upload Image for Analysis")
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload"
    )
    
    if uploaded_image:
        col_img, col_result = st.columns([1, 1])
        
        with col_img:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col_result:
            if st.button("🔍 Analyze Image", key="analyze_img"):
                detector = load_detector()
                if detector:
                    # Save temp file
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(uploaded_image.name)[1]
                    ) as tmp:
                        tmp.write(uploaded_image.getvalue())
                        tmp_path = tmp.name
                    
                    with st.spinner("🔄 Analyzing image..."):
                        result = detector.detect_image(tmp_path)
                    
                    os.unlink(tmp_path)
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        add_to_audit(result)
                        display_result(result)

with tab2:
    st.markdown("### Upload Video for Analysis")
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload"
    )
    
    if uploaded_video:
        st.video(uploaded_video)
        
        if st.button("🔍 Analyze Video", key="analyze_vid"):
            detector = load_detector()
            if detector:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded_video.name)[1]
                ) as tmp:
                    tmp.write(uploaded_video.getvalue())
                    tmp_path = tmp.name
                
                with st.spinner("🔄 Analyzing video frames..."):
                    result = detector.detect_video(tmp_path)
                
                os.unlink(tmp_path)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    add_to_audit(result)
                    display_result(result)
                    
                    # Frame-by-frame scores chart
                    if "frame_scores" in result:
                        st.markdown("#### Frame-by-Frame Analysis")
                        import plotly.graph_objects as go
                        
                        scores = result["frame_scores"]
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=list(range(len(scores))),
                            y=scores,
                            marker_color=[
                                '#ff4444' if s > 0.5 else '#00ff88'
                                for s in scores
                            ]
                        ))
                        fig.add_hline(y=0.5, line_dash="dash",
                                      line_color="yellow",
                                      annotation_text="Threshold")
                        fig.update_layout(
                            title="Fake Probability per Frame",
                            xaxis_title="Frame Index",
                            yaxis_title="Fake Probability",
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            yaxis_range=[0, 1]
                        )
                        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Upload Audio for Analysis")
    uploaded_audio = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "flac", "ogg"],
        key="aud_upload"
    )
    
    if uploaded_audio:
        st.audio(uploaded_audio)
        
        if st.button("🔍 Analyze Audio", key="analyze_aud"):
            detector = load_detector()
            if detector:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]
                ) as tmp:
                    tmp.write(uploaded_audio.getvalue())
                    tmp_path = tmp.name
                
                with st.spinner("🔄 Analyzing audio signal..."):
                    result = detector.detect_audio(tmp_path)
                
                os.unlink(tmp_path)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    add_to_audit(result)
                    display_result(result)


# ─── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #4a5568; font-size: 0.85rem;">'
    '🛡️ DeepfakeGuard — Powered by MobileNetV2 & TensorFlow | '
    'Built for Vulnuris Hackathon 2025'
    '</p>',
    unsafe_allow_html=True
)

