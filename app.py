"""
DeepFaceReal-Physics Streamlit UI v2.0.0
Powered By DeathLegionTeamLK
Professional UI with all engine controls, audio input modes, HeyGen Mode preset,
mobile camera QR, recording export, and dark theme.
"""
import os
import sys
import cv2
import numpy as np
import streamlit as st
import time
import json
from PIL import Image
from io import BytesIO
from typing import Optional
import io
import base64

sys.path.insert(0, os.path.dirname(__file__))

from core.face_swapper import get_swapper
from core.character_manager import get_character_manager, Character
from core.llm_character import create_llm_character
from core.physics_engine import get_tracker
from core.background_engine import get_background
from core.face_3d_engine import get_face_3d_engine
from core.talking_head import get_talking_head
from core.lip_sync import create_lip_sync
from core.eye_engine import get_eye_engine
from core.gesture_engine import get_gesture_engine
from core.pipeline import get_realtime_pipeline, RealTimePipeline

CREDITS = "Powered By DeathLegionTeamLK"
APP_VERSION = "v2.0.0"

st.set_page_config(
    page_title="DeepFaceReal-Physics - Hyper-Realistic AI Avatar",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.2em; font-weight: bold; color: #ff6b6b; margin-bottom: 0; }
    .sub-header { font-size: 1.0em; color: #aaa; margin-top: 0; }
    .char-card { padding: 1em; border: 1px solid #444; border-radius: 10px; margin: 0.5em 0; background: #1a1a2e; }
    .success-text { color: #00ff88; }
    .warning-text { color: #ffaa00; }
    .stats-box { background: #1e1e1e; border-radius: 8px; padding: 12px; margin: 8px 0; }
    .credit-footer { text-align: center; color: #888; padding: 20px 0 10px 0; font-size: 0.9em; border-top: 1px solid #333; margin-top: 30px; }
    .powered { color: #ff6b6b; font-weight: bold; }
    .heygen-badge { background: linear-gradient(135deg, #ff6b6b, #ffa500); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; }
    .engine-card { border: 1px solid #444; border-radius: 8px; padding: 10px; margin: 5px 0; background: #16213e; }
    .stRadio > div { flex-direction: row; gap: 10px; }
    .stRadio label { padding: 8px 16px; border: 1px solid #555; border-radius: 8px; }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

if 'swapper' not in st.session_state:
    with st.spinner("Loading face swap engine..."):
        st.session_state.swapper = get_swapper(det_size=(320, 320))
if 'char_manager' not in st.session_state:
    st.session_state.char_manager = get_character_manager()
if 'llm' not in st.session_state:
    st.session_state.llm = create_llm_character(os.environ.get("OPENROUTER_API_KEY", ""))
if 'face_3d' not in st.session_state:
    st.session_state.face_3d = get_face_3d_engine()
if 'talking_head' not in st.session_state:
    st.session_state.talking_head = get_talking_head()
if 'lip_sync' not in st.session_state:
    st.session_state.lip_sync = create_lip_sync()
if 'eye_engine' not in st.session_state:
    st.session_state.eye_engine = get_eye_engine()
if 'gesture_engine' not in st.session_state:
    st.session_state.gesture_engine = get_gesture_engine()
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = get_realtime_pipeline()

for key, default in [
    ('swap_enabled', True), ('enhance_quality', 'medium'), ('blend_enabled', True),
    ('camera_id', 0), ('source_photo', None), ('source_face', None),
    ('active_character', None), ('pipeline_running', False), ('chat_messages', []),
    ('physics_enabled', True), ('physics_intensity', 0.7), ('background_mode', 'parallax'),
    ('hand_viz_enabled', True), ('eye_engine_enabled', True), ('gesture_enabled', True),
    ('talking_head_enabled', True), ('lip_sync_enabled', True), ('audio_input_mode', 'none'),
    ('heygen_mode', False), ('camera_source', 'usb'), ('camera_url', ''),
    ('recording', False), ('recorded_frames', []), ('show_3d_mesh', False),
    ('show_eye_viz', False), ('show_gesture_viz', False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def load_source_photo(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        with st.spinner("Detecting face in source photo..."):
            source_face = st.session_state.swapper.get_source_face(img)
        if source_face is not None:
            st.session_state.source_photo = img
            st.session_state.source_face = source_face
            return True, f"Face detected! Confidence: {source_face.det_score:.3f}"
        return False, "No face detected in the uploaded photo."
    return False, "No file uploaded."

def apply_heygen_mode():
    st.session_state.heygen_mode = True
    st.session_state.swap_enabled = True
    st.session_state.physics_enabled = True
    st.session_state.eye_engine_enabled = True
    st.session_state.gesture_enabled = True
    st.session_state.talking_head_enabled = True
    st.session_state.lip_sync_enabled = True
    st.session_state.background_mode = 'parallax'
    st.session_state.enhance_quality = 'high'
    st.session_state.blend_enabled = True
    st.session_state.physics_intensity = 0.85
    st.success("🚀 HeyGen Mode activated! All engines enabled at maximum quality.")

def generate_qr_code(url):
    try:
        import qrcode
        qr = qrcode.QRCode(box_size=5, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    except ImportError:
        return None

with st.sidebar:
    st.markdown('<p class="main-header">🎭 DeepFaceReal<br>Physics</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{APP_VERSION} | {CREDITS}</p>', unsafe_allow_html=True)

    if st.button("🚀 HeyGen Mode", type="primary", use_container_width=True):
        apply_heygen_mode()

    st.markdown("---")
    st.subheader("📷 Source")
    cam_id = st.number_input("Camera ID", 0, 10, st.session_state.camera_id)
    if cam_id != st.session_state.camera_id:
        st.session_state.camera_id = cam_id
    st.session_state.swap_enabled = st.toggle("Face Swap", st.session_state.swap_enabled)

    st.subheader("⚙️ Quality")
    st.session_state.enhance_quality = st.select_slider(
        "Enhancement", ["low", "medium", "high"], st.session_state.enhance_quality
    )
    st.session_state.blend_enabled = st.toggle("Seamless Blending", st.session_state.blend_enabled)

    st.subheader("🎨 Engines")
    st.session_state.face_3d.config.enable_pose_estimation = st.toggle("3D Face Engine", True)
    st.session_state.talking_head_enabled = st.toggle("Talking Head", st.session_state.talking_head_enabled)
    st.session_state.lip_sync_enabled = st.toggle("Lip Sync", st.session_state.lip_sync_enabled)
    st.session_state.eye_engine_enabled = st.toggle("Eye & Gaze", st.session_state.eye_engine_enabled)
    st.session_state.gesture_enabled = st.toggle("Gestures", st.session_state.gesture_enabled)

    st.subheader("🔬 Physics")
    st.session_state.physics_enabled = st.toggle("Physics", st.session_state.physics_enabled)
    st.session_state.physics_intensity = st.slider("Intensity", 0.0, 1.0, st.session_state.physics_intensity, 0.05)

    st.subheader("🎨 Background")
    st.session_state.background_mode = st.selectbox(
        "Mode", ["static", "parallax", "blur"],
        index=["static", "parallax", "blur"].index(st.session_state.background_mode)
    )

    st.subheader("🎵 Audio Input")
    st.session_state.audio_input_mode = st.selectbox(
        "Audio Source", ["none", "mic", "file", "tts"],
        index=["none", "mic", "file", "tts"].index(st.session_state.audio_input_mode)
    )

    st.subheader("📊 Performance")
    pipeline = st.session_state.pipeline
    metrics = pipeline.get_metrics()
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("FPS", f"{metrics.fps:.1f}" if metrics.fps > 0 else "N/A")
    with col_b:
        st.metric("Frames", metrics.frames_processed)

    st.markdown("---")
    st.markdown(
        f'<div style="text-align:center;color:#888;font-size:0.8em;">'
        f'<span class="powered">{CREDITS}</span></div>',
        unsafe_allow_html=True
    )

tabs = st.tabs(["🎯 Avatar Studio", "📱 Mobile", "👤 Characters", "💬 Chat", "🎬 Engines", "⚙️ Settings"])

with tabs[0]:
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("### 📸 Source Photo")
        uploaded = st.file_uploader("Upload face photo", type=['jpg', 'jpeg', 'png'], key="src_photo")
        if uploaded:
            ok, msg = load_source_photo(uploaded)
            if ok:
                st.success(msg)
                st.image(cv2.cvtColor(st.session_state.source_photo, cv2.COLOR_BGR2RGB), width=280)
            else:
                st.error(msg)

        st.markdown("---")
        st.markdown("### 🎯 Preview")
        prev = st.empty()
        if st.session_state.source_photo is not None:
            img = st.session_state.source_photo.copy()
            if st.session_state.swap_enabled and st.session_state.source_face:
                rendered = st.session_state.swapper.process_frame(
                    img, st.session_state.source_face,
                    blend=st.session_state.blend_enabled,
                    enhance=st.session_state.enhance_quality != 'low',
                    enhance_quality=st.session_state.enhance_quality
                )
                prev.image(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB), width=400)
            else:
                prev.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=400)
        else:
            prev.info("Upload a source photo")

    with col2:
        st.markdown("### 🖥️ Real-Time Preview")
        preview_rt = st.empty()
        preview_rt.info("Start the pipeline to see real-time output")

        if st.session_state.heygen_mode:
            st.markdown(
                '<div class="heygen-badge" style="display:inline-block;">🚀 HeyGen Mode Active</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### 💾 Recording")
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            if st.button("⏺ Start Recording" if not st.session_state.recording else "⏹ Stop Recording",
                         type="primary", use_container_width=True):
                st.session_state.recording = not st.session_state.recording
                if st.session_state.recording:
                    st.session_state.recorded_frames = []
                    st.toast("Recording started!")
                else:
                    if st.session_state.recorded_frames:
                        fps = 20
                        out_path = "recorded_output.mp4"
                        h, w = st.session_state.recorded_frames[0].shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                        for f in st.session_state.recorded_frames:
                            out.write(f)
                        out.release()
                        st.success(f"Saved: {out_path}")
                        st.session_state.recorded_frames = []
        with rec_col2:
            st.metric("Recorded Frames", len(st.session_state.recorded_frames))

with tabs[1]:
    st.markdown("### 📱 Mobile Camera Setup")
    st.info("Use your phone as a webcam via IP Webcam app.")

    col_qr, col_guide = st.columns([1, 2])
    with col_qr:
        phone_ip = st.text_input("Phone IP", placeholder="192.168.1.100")
        if phone_ip:
            mjpeg_url = f"http://{phone_ip}:8080/video"
            st.code(mjpeg_url, language="text")
            qr = generate_qr_code(mjpeg_url)
            if qr:
                st.image(qr, caption="Scan to connect", width=200)
        if st.button("🔍 Auto-Detect"):
            with st.spinner("Scanning..."):
                from core.webcam_pipeline import scan_ip_webcam_urls
                found = scan_ip_webcam_urls(timeout=0.5)
                if found:
                    st.success(f"Found: {found[0]}")
                    qr2 = generate_qr_code(found[0])
                    if qr2:
                        st.image(qr2, width=200)
                else:
                    st.warning("Not found. Enter IP manually.")

    with col_guide:
        st.markdown("""
        **Setup Steps:**
        1. Install **IP Webcam** app from Google Play Store
        2. Open app → tap **Start server**
        3. Note IP (e.g., `192.168.1.100:8080`)
        4. Enter IP above or scan QR code

        **Camera Source:**
        - `usb` - Standard webcam
        - `ip_webcam` - Phone camera
        - `custom_url` - Any MJPEG URL
        """)
        st.session_state.camera_source = st.selectbox(
            "Source", ["usb", "ip_webcam", "custom_url"],
            key="cam_src"
        )
        if st.button("Apply Source"):
            st.success(f"Camera source: {st.session_state.camera_source}")

with tabs[2]:
    st.markdown("### 👤 Characters")
    ctabs = st.tabs(["Gallery", "Create", "Manage"])
    with ctabs[0]:
        chars = st.session_state.char_manager.list_characters()
        if chars:
            for name in chars:
                char = st.session_state.char_manager.get_character(name)
                if char:
                    st.markdown(f'<div class="char-card">', unsafe_allow_html=True)
                    st.markdown(f"**{char.name}**")
                    st.caption(f"Voice: {'✅' if char.voice_enabled else '❌'} | Quality: {char.enhance_quality}")
                    if st.button(f"Select {name}", key=f"sel_{name}"):
                        st.session_state.active_character = name
                        st.session_state.llm.set_character(char.name, char.system_prompt)
                        if char.photo_path and os.path.exists(char.photo_path):
                            img = cv2.imread(char.photo_path)
                            st.session_state.source_photo = img
                            face = st.session_state.swapper.get_source_face(img)
                            if face:
                                st.session_state.source_face = face
                        st.success(f"Activated: {name}")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No characters. Create one!")

        if st.session_state.active_character:
            st.success(f"Active: {st.session_state.active_character}")
        else:
            st.warning("No character selected")

    with ctabs[1]:
        st.markdown("### Create Character")
        new_name = st.text_input("Name", placeholder="e.g., Einstein")
        new_prompt = st.text_area("System Prompt",
                                  "You are a friendly assistant. Be helpful and concise.", height=100)
        new_voice = st.toggle("Voice", False)
        char_photo = st.file_uploader("Photo", type=['jpg', 'jpeg', 'png'], key="char_photo")
        if st.button("Create", type="primary"):
            if new_name:
                char = st.session_state.char_manager.create_character(
                    name=new_name, system_prompt=new_prompt, voice_enabled=new_voice
                )
                if char_photo:
                    path = os.path.join(os.path.dirname(__file__), 'profiles',
                                        f"{new_name.replace(' ', '_')}_photo.jpg")
                    with open(path, 'wb') as f:
                        f.write(char_photo.getvalue())
                    nparr = np.frombuffer(char_photo.getvalue(), np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    char.photo_path = path
                    face_emb = st.session_state.swapper.extract_face_embedding(img)
                    src_face = st.session_state.swapper.get_source_face(img)
                    if face_emb is not None:
                        char.face_embedding = face_emb
                    if src_face is not None:
                        char.source_face = src_face
                st.session_state.char_manager.save_character(new_name)
                st.success(f"Character '{new_name}' created!")
                st.rerun()
            else:
                st.error("Enter a name")

    with ctabs[2]:
        chars = st.session_state.char_manager.list_characters()
        if chars:
            del_name = st.selectbox("Select to delete", chars, key="del_char")
            if st.button("🗑️ Delete"):
                st.session_state.char_manager.delete_character(del_name)
                if st.session_state.active_character == del_name:
                    st.session_state.active_character = None
                st.success(f"Deleted: {del_name}")
                st.rerun()
        else:
            st.info("No characters to manage")

with tabs[3]:
    st.markdown("### 💬 Chat with Character")
    chat = st.container(height=300)
    with chat:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    if prompt := st.chat_input("Type your message..."):
        if st.session_state.active_character is None:
            st.warning("Select a character first")
        else:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.llm.send_message(prompt)
                st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
    if st.button("Clear Chat"):
        st.session_state.chat_messages = []
        st.session_state.llm.clear_history()
        st.rerun()

with tabs[4]:
    st.markdown("### 🎬 Engine Status")
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.markdown('<div class="engine-card">', unsafe_allow_html=True)
        f3d = st.session_state.face_3d.get_status()
        st.markdown(f"**3D Face Engine**")
        st.caption(f"FPS: {f3d['fps']} | Frames: {f3d['frames_processed']}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="engine-card">', unsafe_allow_html=True)
        th = st.session_state.talking_head.get_status()
        st.markdown(f"**Talking Head**")
        st.caption(f"Librosa: {'✅' if th['has_librosa'] else '❌'} | Frames: {th['frame_count']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with ec2:
        st.markdown('<div class="engine-card">', unsafe_allow_html=True)
        ls = st.session_state.lip_sync.get_status()
        st.markdown(f"**Lip Sync**")
        st.caption(f"Model: {ls['model_type']} | Wav2Lip: {'✅' if ls['wav2lip_available'] else '❌'} | Phoneme: {ls['last_phoneme']}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="engine-card">', unsafe_allow_html=True)
        ee = st.session_state.eye_engine.get_status()
        st.markdown(f"**Eye & Gaze**")
        st.caption(f"Blink: {ee['state']['is_blinking']} | Gaze: {ee['state']['gaze']} | Pupil: {ee['state']['pupil_size']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with ec3:
        st.markdown('<div class="engine-card">', unsafe_allow_html=True)
        ge = st.session_state.gesture_engine.get_status()
        st.markdown(f"**Gesture Engine**")
        st.caption(f"Type: {ge['state']['gesture_type']} | Queue: {ge['state']['queue_size']}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="engine-card">', unsafe_allow_html=True)
        pm = st.session_state.pipeline.get_metrics()
        st.markdown(f"**Pipeline**")
        st.caption(f"FPS: {pm.fps:.1f} | Frames: {pm.frames_processed} | Latency: {pm.average_latency:.1f}ms")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Per-Engine Stage Times")
    if pm and pm.stage_times:
        stages = pm.stage_times
        cols = st.columns(len(stages))
        for i, (name, time_ms) in enumerate(stages.items()):
            with cols[i]:
                st.metric(name, f"{time_ms:.1f}ms")

with tabs[5]:
    st.markdown("### ⚙️ Settings")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("#### Face Swap")
        st.session_state.swap_enabled = st.toggle("Enabled", st.session_state.swap_enabled, key="ss1")
        st.session_state.blend_enabled = st.toggle("Blending", st.session_state.blend_enabled, key="ss2")
        st.session_state.enhance_quality = st.select_slider(
            "Quality", ["low", "medium", "high"], st.session_state.enhance_quality, key="ss3"
        )

        st.markdown("#### Physics")
        st.session_state.physics_enabled = st.toggle("Physics Enabled", st.session_state.physics_enabled, key="ss4")
        st.session_state.physics_intensity = st.slider("Intensity", 0.0, 1.0, st.session_state.physics_intensity, 0.05, key="ss5")

        st.markdown("#### Background")
        st.session_state.background_mode = st.selectbox("Mode", ["static", "parallax", "blur"],
                                                        key="ss6")

    with s2:
        st.markdown("#### Engines")
        st.session_state.eye_engine_enabled = st.toggle("Eye Engine", st.session_state.eye_engine_enabled, key="ss7")
        st.session_state.gesture_enabled = st.toggle("Gesture Engine", st.session_state.gesture_enabled, key="ss8")
        st.session_state.talking_head_enabled = st.toggle("Talking Head", st.session_state.talking_head_enabled, key="ss9")
        st.session_state.lip_sync_enabled = st.toggle("Lip Sync", st.session_state.lip_sync_enabled, key="ss10")

        st.markdown("#### Visualization")
        st.session_state.show_3d_mesh = st.toggle("Show 3D Mesh", st.session_state.show_3d_mesh, key="ss11")
        st.session_state.show_eye_viz = st.toggle("Show Eye Tracking", st.session_state.show_eye_viz, key="ss12")
        st.session_state.show_gesture_viz = st.toggle("Show Gesture Info", st.session_state.show_gesture_viz, key="ss13")

    st.markdown("---")
    st.markdown("#### System Info")
    st.code(
        f"Version: {APP_VERSION}\n"
        f"Physics: {'Active' if st.session_state.physics_enabled else 'Disabled'}\n"
        f"3D Face: {'Active' if st.session_state.face_3d.config.enable_pose_estimation else 'Disabled'}\n"
        f"Talking Head: {'Active' if st.session_state.talking_head_enabled else 'Disabled'}\n"
        f"Lip Sync: {'Active' if st.session_state.lip_sync_enabled else 'Disabled'}\n"
        f"Eye Engine: {'Active' if st.session_state.eye_engine_enabled else 'Disabled'}\n"
        f"Gestures: {'Active' if st.session_state.gesture_enabled else 'Disabled'}\n"
        f"Background: {st.session_state.background_mode}\n"
        f"Audio: {st.session_state.audio_input_mode}\n"
        f"HeyGen Mode: {'ON' if st.session_state.heygen_mode else 'OFF'}\n"
        f"Credits: {CREDITS}",
        language="text"
    )

st.markdown("---")
st.markdown(
    f'<div class="credit-footer">'
    f'🎭 DeepFaceReal-Physics {APP_VERSION} | Hyper-Realistic AI Avatar | '
    f'<span class="powered">{CREDITS}</span>'
    f'</div>',
    unsafe_allow_html=True
)