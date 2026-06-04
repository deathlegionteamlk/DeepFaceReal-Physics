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

sys.path.insert(0, os.path.dirname(__file__))

from core.face_swapper import get_swapper, FaceSwapper
from core.character_manager import get_character_manager, Character, CharacterManager
from core.llm_character import create_llm_character, LLMCharacter
from core.physics_engine import get_tracker, HolisticTracker, CREDITS as PHY_CREDITS
from core.background_engine import get_background, ParallaxBackground

CREDITS = 'Powered By DeathLegionTeamLK'

st.set_page_config(
    page_title="DeepFaceReal Physics - Real-Time Deepfake",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header { font-size: 2em; font-weight: bold; margin-bottom: 0; }
.char-card { padding: 1em; border: 1px solid #444; border-radius: 10px; margin: 0.5em 0; }
.success-text { color: #00ff88; }
.warning-text { color: #ffaa00; }
.stChat { margin-top: 1em; }
.stats-box { background: #1e1e1e; border-radius: 8px; padding: 12px; margin: 8px 0; }
.credit-footer { text-align: center; color: #888; padding: 20px 0 10px 0; font-size: 0.9em; border-top: 1px solid #333; margin-top: 30px; }
.powered { color: #ff6b6b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

if 'swapper' not in st.session_state:
    with st.spinner("Loading face swap engine..."):
        st.session_state.swapper = get_swapper(det_size=(320, 320))

if 'char_manager' not in st.session_state:
    st.session_state.char_manager = get_character_manager()

if 'llm' not in st.session_state:
    st.session_state.llm = create_llm_character(os.environ.get("OPENROUTER_API_KEY", ""))

if 'swap_enabled' not in st.session_state:
    st.session_state.swap_enabled = True

if 'enhance_quality' not in st.session_state:
    st.session_state.enhance_quality = "medium"

if 'blend_enabled' not in st.session_state:
    st.session_state.blend_enabled = True

if 'camera_id' not in st.session_state:
    st.session_state.camera_id = 0

if 'source_photo' not in st.session_state:
    st.session_state.source_photo = None

if 'source_face' not in st.session_state:
    st.session_state.source_face = None

if 'active_character' not in st.session_state:
    st.session_state.active_character = None

if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

if 'physics_enabled' not in st.session_state:
    st.session_state.physics_enabled = True

if 'physics_intensity' not in st.session_state:
    st.session_state.physics_intensity = 0.7

if 'background_mode' not in st.session_state:
    st.session_state.background_mode = 'parallax'

if 'hand_viz_enabled' not in st.session_state:
    st.session_state.hand_viz_enabled = True

if 'tracker' not in st.session_state:
    st.session_state.tracker = get_tracker()

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
        else:
            return False, "No face detected in the uploaded photo. Try a different image."
    return False, "No file uploaded."

def process_frame_callback(frame, source_face):
    swapper = st.session_state.swapper
    swapper.physics_enabled = st.session_state.physics_enabled
    swapper.hand_overlay_enabled = st.session_state.hand_viz_enabled
    swapper.background_mode = st.session_state.background_mode
    if st.session_state.physics_enabled and hasattr(swapper, 'tracker'):
        swapper.tracker.config.intensity = st.session_state.physics_intensity
    return swapper.process_frame(
        frame,
        source_face,
        blend=st.session_state.blend_enabled,
        enhance=(st.session_state.enhance_quality != 'low'),
        enhance_quality=st.session_state.enhance_quality
    )

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
    st.markdown('<p class="main-header">🎭 DeepFaceReal Physics</p>', unsafe_allow_html=True)
    st.caption(f'{CREDITS}')
    st.markdown("---")

    st.subheader("📷 Camera")
    col1, col2 = st.columns(2)
    with col1:
        camera_id = st.number_input("Camera ID", min_value=0, max_value=10, value=st.session_state.camera_id)
        if camera_id != st.session_state.camera_id:
            st.session_state.camera_id = camera_id
    with col2:
        st.session_state.swap_enabled = st.toggle("Swap Active", value=st.session_state.swap_enabled)

    st.subheader("⚙️ Quality")
    st.session_state.enhance_quality = st.select_slider(
        "Enhancement", options=["low", "medium", "high"], value=st.session_state.enhance_quality
    )
    st.session_state.blend_enabled = st.toggle("Seamless Blending", value=st.session_state.blend_enabled)

    st.subheader("🔬 Physics")
    st.session_state.physics_enabled = st.toggle("Physics Engine", value=st.session_state.physics_enabled)
    st.session_state.physics_intensity = st.slider(
        "Physics Intensity", 0.0, 1.0, st.session_state.physics_intensity, 0.05
    )

    st.subheader("🎨 Background")
    st.session_state.background_mode = st.selectbox(
        "Background Mode", ["static", "parallax", "blur"],
        index=["static", "parallax", "blur"].index(st.session_state.background_mode)
    )

    st.subheader("✋ Hand Tracking")
    st.session_state.hand_viz_enabled = st.toggle("Show Hand Skeleton", value=st.session_state.hand_viz_enabled)

    st.subheader("📊 Physics Status")
    tracker = st.session_state.get('tracker')
    if tracker:
        overlay_data = tracker.get_landmark_data_for_overlay()
        lmk_count = overlay_data.get('landmark_count', 0)
        fps = overlay_data.get('fps', 0)
        st.metric("Landmarks Tracked", lmk_count)
        st.metric("Tracking FPS", f"{fps:.1f}")
        blink_state = overlay_data.get('blink_state', 0)
        ear = overlay_data.get('ear_value', 0)
        st.metric("Blink State", f"{blink_state:.2f}")
        st.caption(f"EAR: {ear:.3f}")

    st.subheader("🎬 Pipeline")
    if st.button("▶️ Start Pipeline" if not st.session_state.pipeline_running else "⏹️ Stop Pipeline"):
        st.session_state.pipeline_running = not st.session_state.pipeline_running
        if st.session_state.pipeline_running:
            st.info("Pipeline started (simulated mode - webcam not available in cloud)")
        else:
            st.info("Pipeline stopped")
    st.markdown("---")

    with st.expander("📖 Quick Guide", expanded=False):
        st.markdown("""
1. **Upload a source photo** with a face
2. **Select or create a character**
3. **Toggle swap ON** to see the face swap
4. **Chat** with your character via LLM
5. **Enable Physics** for full body tracking & smooth motion
6. **Try Parallax** for 3D depth background

Features:
- **InsightFace** for face detection & swapping
- **MediaPipe Holistic** full body tracking (543 landmarks)
- **Physics Engine** with inertia, springs & momentum
- **Parallax Background** 3-layer depth system
- **Poisson blending** for seamless compositing
- **OpenRouter** for LLM personality
""")

    st.markdown("---")
    st.markdown(
        f'<div class="credit-footer">🎭 <span class="powered">{CREDITS}</span></div>',
        unsafe_allow_html=True
    )

main_tabs = st.tabs(["🎯 Face Swap", "📱 Mobile Camera", "👤 Characters", "💬 Chat", "⚙️ Settings"])

with main_tabs[0]:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<p class="main-header">🎯 Source Photo</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a face photo to swap onto your webcam feed",
            type=['jpg', 'jpeg', 'png'], key="source_photo_uploader"
        )
        if uploaded_file:
            success, msg = load_source_photo(uploaded_file)
            if success:
                st.success(msg)
                img_rgb = cv2.cvtColor(st.session_state.source_photo, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="Source Photo", width=300)
            else:
                st.error(msg)

        st.markdown("---")
        st.markdown('<p class="main-header">🖼️ Preview</p>', unsafe_allow_html=True)
        preview_placeholder = st.empty()

        if st.session_state.source_photo is not None:
            img_copy = st.session_state.source_photo.copy()
            if st.session_state.swap_enabled and st.session_state.source_face:
                rendered = st.session_state.swapper.process_frame(
                    img_copy, st.session_state.source_face,
                    blend=True, enhance=(st.session_state.enhance_quality != 'low'),
                    enhance_quality=st.session_state.enhance_quality
                )
                img_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(img_rgb, caption="Swap Preview", width=400)
        else:
            preview_placeholder.info("Upload a source photo to see preview")

        st.markdown("---")
        status_cols = st.columns(4)
        with status_cols[0]:
            st.metric("Swap", "🟢 Active" if st.session_state.swap_enabled else "🔴 Off")
        with status_cols[1]:
            st.metric("Enhance", st.session_state.enhance_quality.title())
        with status_cols[2]:
            st.metric("Blending", "✅ On" if st.session_state.blend_enabled else "❌ Off")
        with status_cols[3]:
            st.metric("Physics", "✅ On" if st.session_state.physics_enabled else "❌ Off")

with main_tabs[1]:
    st.markdown('<p class="main-header">📱 Mobile Camera via IP Webcam</p>', unsafe_allow_html=True)
    st.markdown("### Use your phone as a webcam for WhatsApp calls")
    st.info("This feature lets you use your phone's camera as the video source for face swapping, enabling WhatsApp Mobile integration.")

    col_qr, col_guide = st.columns([1, 2])

    with col_qr:
        st.markdown("#### Connect your Phone")
        phone_ip = st.text_input("Phone IP Address", placeholder="e.g., 192.168.1.100")
        if phone_ip:
            mjpeg_url = f"http://{phone_ip}:8080/video"
            st.code(mjpeg_url, language="text")
            qr_bytes = generate_qr_code(mjpeg_url)
            if qr_bytes:
                st.image(qr_bytes, caption="Scan to connect", width=200)
            st.markdown("**Or use auto-detect:**")
            if st.button("🔍 Scan Local Network"):
                with st.spinner("Scanning for IP Webcams..."):
                    from core.webcam_pipeline import scan_ip_webcam_urls
                    found = scan_ip_webcam_urls(timeout=0.5)
                    if found:
                        st.success(f"Found: {found[0]}")
                        qr_bytes2 = generate_qr_code(found[0])
                        if qr_bytes2:
                            st.image(qr_bytes2, width=200)
                    else:
                        st.warning("No IP Webcams found. Enter IP manually.")

    with col_guide:
        st.markdown("#### Setup Instructions")
        st.markdown("""
**For Android (IP Webcam app):**
1. Install **IP Webcam** from Google Play Store
2. Open the app and scroll to **Start server**
3. Note the IP address shown (e.g., `http://192.168.1.100:8080`)
4. Enter the IP above or scan the QR code
5. The app streams your phone camera to the face swap pipeline

**For WhatsApp Desktop:**
1. Start the pipeline with IP Webcam source
2. Open WhatsApp Desktop
3. In call settings, select **DeepFakeCam** as camera
4. Your swapped face appears in real-time

**Camera Source Options:**
- `usb` - Standard USB webcam
- `ip_webcam` - Phone camera via IP Webcam app
- `custom_url` - Any MJPEG stream URL
        """)

    st.markdown("---")
    st.markdown("#### Camera Source Configuration")
    cam_source = st.selectbox("Camera Source", ["usb", "ip_webcam", "custom_url"])
    cam_url = st.text_input("Custom URL (if applicable)", placeholder="http://192.168.1.100:8080/video")
    if st.button("Apply Camera Source"):
        st.session_state.camera_source = cam_source
        st.session_state.camera_url = cam_url
        st.success(f"Camera source set to: {cam_source}")

with main_tabs[2]:
    st.markdown('<p class="main-header">👤 Characters</p>', unsafe_allow_html=True)
    char_tabs = st.tabs(["Gallery", "Create", "Manage"])

    with char_tabs[0]:
        characters = st.session_state.char_manager.list_characters()
        if characters:
            for char_name in characters:
                char = st.session_state.char_manager.get_character(char_name)
                if char:
                    with st.container():
                        st.markdown(f'<div class="char-card">', unsafe_allow_html=True)
                        st.markdown(f"**{char.name}**")
                        st.caption(f"Voice: {'✅' if char.voice_enabled else '❌'} | Swap: {'✅' if char.swap_enabled else '❌'} | Quality: {char.enhance_quality}")
                        if st.button(f"Select {char.name}", key=f"select_{char_name}"):
                            st.session_state.active_character = char_name
                            st.session_state.llm.set_character(char.name, char.system_prompt)
                            if char.photo_path and os.path.exists(char.photo_path):
                                img = cv2.imread(char.photo_path)
                                st.session_state.source_photo = img
                                face = st.session_state.swapper.get_source_face(img)
                                if face:
                                    st.session_state.source_face = face
                            st.success(f"Activated: {char.name}")
                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No characters yet. Create one in the 'Create' tab.")

        if st.session_state.active_character:
            st.success(f"Active: {st.session_state.active_character}")
        else:
            st.warning("No character selected")

    with char_tabs[1]:
        st.markdown("### Create New Character")
        new_name = st.text_input("Character Name", placeholder="e.g., Einstein")
        new_prompt = st.text_area(
            "System Prompt (personality)",
            value="You are a friendly and knowledgeable assistant. Be helpful and concise.",
            height=150
        )
        new_voice = st.toggle("Enable Voice", value=False)
        char_photo = st.file_uploader("Character Photo (optional)", type=['jpg', 'jpeg', 'png'], key="char_photo_uploader")

        if st.button("Create Character", type="primary"):
            if new_name:
                char = st.session_state.char_manager.create_character(
                    name=new_name, system_prompt=new_prompt, voice_enabled=new_voice
                )
                if char_photo is not None:
                    photo_path = os.path.join(
                        os.path.dirname(__file__), 'profiles',
                        f"{new_name.replace(' ', '_')}_photo.jpg"
                    )
                    with open(photo_path, 'wb') as f:
                        f.write(char_photo.getvalue())
                    bytes_data = char_photo.getvalue()
                    nparr = np.frombuffer(bytes_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    face_emb = st.session_state.swapper.extract_face_embedding(img)
                    source_face = st.session_state.swapper.get_source_face(img)
                    char.photo_path = photo_path
                    if face_emb is not None:
                        char.face_embedding = face_emb
                    if source_face is not None:
                        char.source_face = source_face
                st.session_state.char_manager.save_character(new_name)
                st.success(f"Character '{new_name}' created!")
                st.rerun()
            else:
                st.error("Please enter a character name")

    with char_tabs[2]:
        characters = st.session_state.char_manager.list_characters()
        if characters:
            delete_name = st.selectbox("Select character to delete", characters, key="delete_select")
            if st.button("🗑️ Delete", type="secondary"):
                st.session_state.char_manager.delete_character(delete_name)
                if st.session_state.active_character == delete_name:
                    st.session_state.active_character = None
                st.success(f"Deleted: {delete_name}")
                st.rerun()
        else:
            st.info("No characters to manage.")

with main_tabs[3]:
    st.markdown("### 💬 Chat with Character")
    chat_container = st.container(height=300)
    with chat_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Type your message..."):
        if st.session_state.active_character is None:
            st.warning("Please select a character first")
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

with main_tabs[4]:
    st.markdown('<p class="main-header">⚙️ Settings</p>', unsafe_allow_html=True)
    st.markdown("#### Face Swap Settings")
    st.session_state.swap_enabled = st.toggle("Face Swap Enabled", value=st.session_state.swap_enabled, key="settings_swap")
    st.session_state.blend_enabled = st.toggle("Seamless Blending (Poisson)", value=st.session_state.blend_enabled, key="settings_blend")
    st.session_state.enhance_quality = st.select_slider(
        "Enhancement Quality", options=["low", "medium", "high"], value=st.session_state.enhance_quality, key="settings_enhance"
    )

    st.markdown("#### Physics Settings")
    st.session_state.physics_enabled = st.toggle("Physics Engine Enabled", value=st.session_state.physics_enabled, key="settings_physics")
    st.session_state.physics_intensity = st.slider(
        "Physics Intensity", 0.0, 1.0, st.session_state.physics_intensity, 0.05, key="settings_phys_intensity"
    )

    st.markdown("#### Background Settings")
    st.session_state.background_mode = st.selectbox(
        "Background Mode", ["static", "parallax", "blur"],
        index=["static", "parallax", "blur"].index(st.session_state.background_mode),
        key="settings_bg"
    )

    st.markdown("#### Hand Tracking")
    st.session_state.hand_viz_enabled = st.toggle("Show Hand Skeleton Overlay", value=st.session_state.hand_viz_enabled, key="settings_hands")

    st.markdown("#### System Info")
    st.code(f"Physics Engine: {'Active' if st.session_state.physics_enabled else 'Disabled'}\nBackground Mode: {st.session_state.background_mode}\nHand Viz: {'On' if st.session_state.hand_viz_enabled else 'Off'}\nCredits: {CREDITS}", language="text")

st.markdown("---")
st.markdown(
    f'<div class="credit-footer">'
    f'🎭 DeepFaceReal Physics | Full Body Tracking + Physics + WhatsApp Mobile/Desktop | '
    f'<span class="powered">{CREDITS}</span>'
    f'</div>',
    unsafe_allow_html=True
)