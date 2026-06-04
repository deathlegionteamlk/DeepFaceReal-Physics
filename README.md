[![GitHub Stars](https://img.shields.io/github/stars/deathlegionteamlk/DeepFaceReal-Physics?style=social)](https://github.com/deathlegionteamlk/DeepFaceReal-Physics)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![GitHub Release](https://img.shields.io/github/v/release/deathlegionteamlk/DeepFaceReal-Physics)](https://github.com/deathlegionteamlk/DeepFaceReal-Physics/releases)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/deathlegionteamlk/DeepFaceReal-Physics/graphs/commit-activity)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/Built%20with-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

<meta name="description" content="DeepFaceReal-Physics — Hyper-Realistic AI Avatar System with HeyGen-Level Realism. 3D face reconstruction, audio-driven talking head, Wav2Lip lip sync, natural eye gaze, conversational gestures, and full body physics simulation. CPU-optimized for real-time performance. Powered By DeathLegionTeamLK.">
<meta name="keywords" content="ai avatar, talking head, deepfake, 3d face reconstruction, wav2lip, lip sync, mediapipe, face swap, heygen alternative, sadtalker, real-time avatar, conversational ai, cpu optimized">
<meta name="author" content="DeathLegionTeamLK">

<div align="center">
<h1>🎭 DeepFaceReal-Physics</h1>
<h2>Hyper-Realistic AI Avatar with HeyGen-Level Realism</h2>
<p><strong>The most advanced open-source AI avatar system — 3D face reconstruction, audio-driven talking head, Wav2Lip lip sync, natural eye gaze, conversational gestures, and real-time pipeline optimization on CPU.</strong></p>
<p><em>Powered By <strong>DeathLegionTeamLK</strong></em></p>

<br>
<p>
<a href="#-features"><strong>Features</strong></a> •
<a href="#-comparison"><strong>HeyGen vs SadTalker vs DeepFaceReal</strong></a> •
<a href="#-quick-start"><strong>Quick Start</strong></a> •
<a href="#-architecture"><strong>Architecture</strong></a> •
<a href="#-engine-documentation"><strong>Engine Docs</strong></a> •
<a href="#-api-v2"><strong>API v2</strong></a> •
<a href="#%EF%B8%8F-build-windows-exe"><strong>Windows EXE</strong></a> •
<a href="#-credits"><strong>Credits</strong></a>
</p>
<br>

[🌟 Star on GitHub](https://github.com/deathlegionteamlk/DeepFaceReal-Physics) • [🐛 Report Bug](https://github.com/deathlegionteamlk/DeepFaceReal-Physics/issues) • [📖 Read Docs](https://github.com/deathlegionteamlk/DeepFaceReal-Physics/blob/main/README.md)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **3D Face Engine** | 468 MediaPipe face mesh landmarks, Delaunay triangulation for 3D surface mapping, 6DoF head pose estimation (pitch/yaw/roll/xyz), expression coefficient extraction |
| 🗣️ **Audio-Driven Talking Head** | MFCC/pitch/energy feature extraction, head pose prediction from audio rhythm, speech-synchronized expressions, natural nodding/tilting patterns (inspired by SadTalker/Ditto) |
| 👄 **Wav2Lip Lip Sync** | Phoneme detection from audio, lip shape prediction matching spoken phonemes, Wav2Lip inference on face region, temporal smoothness, audio buffering for real-time streaming |
| 👁️ **Natural Eye & Gaze** | Saccade simulation (200-300ms intervals), micro-saccades during fixation, natural blink patterns (every 2-4s, 100-400ms duration), gaze target tracking, pupil rendering |
| 👐 **Conversational Gestures** | Speech-rhythm hand gestures, shoulder/head micro-movements, posture variations, configurable gesture intensity |
| 🔄 **Real-Time Pipeline** | Async pipeline: Audio→Features→TalkingHead→LipSync→EyeGaze→Gestures→Composite. Frame skipping, cached inference, async queues for CPU optimization |
| 🧠 **Full Body Physics** | MediaPipe Holistic (543 landmarks), momentum/inertia, spring dynamics, blink detection |
| 🖼️ **Parallax Background** | 3-layer depth parallax driven by head position, depth-based blur |
| 📱 **WhatsApp Mobile** | IP Webcam integration for phone-as-camera |
| 💬 **LLM Characters** | OpenRouter-powered character AI with personality system prompts |
| 🖥️ **Professional UI** | Streamlit v2 dashboard with all engine controls, Heygen Mode preset, recording export |
| 🔌 **REST API + WebSocket** | FastAPI v2 with /generate/talking-head, /animate/face, real-time streaming |
| 🪟 **Windows EXE** | Build standalone DeepFaceReal.exe with PyInstaller |

## 📊 Comparison

| Capability | **HeyGen** ($24-240/mo) | **SadTalker** | **DeepFaceReal-Physics** |
|------------|:---:|:---:|:---:|
| 3D Face Reconstruction | ✅ | ❌ | ✅ |
| Audio-Driven Head Motion | ✅ | ✅ | ✅ |
| Wav2Lip Lip Sync | ✅ | ❌ | ✅ |
| Natural Eye Gaze | ✅ | ❌ | ✅ |
| Conversational Gestures | ✅ | ❌ | ✅ |
| Real-Time (≥15 FPS CPU) | ❌ (Cloud) | ⚠️ (10 FPS GPU) | ✅ (15-20 FPS CPU) |
| Face Swap | ❌ | ❌ | ✅ |
| LLM Character AI | ⚠️ (Limited) | ❌ | ✅ |
| Self-Hosted | ❌ | ✅ | ✅ |
| Open Source | ❌ | ✅ | ✅ |
| Price | **$24-240/month** | Free | **Free** |
| WhatsApp Integration | ❌ | ❌ | ✅ |
| Windows EXE | N/A | Manual | ✅ (PyInstaller) |
| API + WebSocket | ✅ | ❌ | ✅ |
| GPU Required | ✅ (Cloud) | ✅ (GPU) | ❌ **(CPU only)** |

> **DeepFaceReal-Physics matches or exceeds HeyGen features while being completely free, open-source, and CPU-optimized.** No GPU required, no monthly subscription.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 4GB+ RAM (8GB recommended)
- Webcam (USB or Android phone via IP Webcam)

### Installation

```bash
# Clone the repository
git clone https://github.com/deathlegionteamlk/DeepFaceReal-Physics.git
cd DeepFaceReal-Physics

# Install dependencies
pip install -r requirements.txt

# Run the web dashboard
streamlit run app.py --server.port 8080

# In another terminal, run the API server
python api.py
```

### Docker

```bash
docker build -t deepfacereal-physics .
docker run -p 8080:8080 -p 8081:8081 deepfacereal-physics
```

## 🏗️ Architecture

```
                          ┌──────────────────────────────┐
                          │    Input Sources              │
                          │  ┌─────┐ ┌────────┐ ┌──────┐ │
                          │  │USB  │ │IP      │ │Audio │ │
                          │  │Cam  │ │Webcam  │ │File  │ │
                          │  └──┬──┘ └───┬────┘ └──┬───┘ │
                          └─────┼────────┼─────────┼─────┘
                                │        │         │
                          ┌─────▼────────▼─────────▼──────┐
                          │    Audio Feature Extraction    │
                          │  (MFCC, Pitch, Energy, F0)    │
                          └───────────┬───────────────────┘
                                      │
                          ┌───────────▼───────────────────┐
                          │  3D Face Engine               │
                          │  ┌─────────────────────────┐  │
                          │  │ MediaPipe Face Mesh     │  │
                          │  │ (468 landmarks)         │  │
                          │  │ Delaunay Triangulation  │  │
                          │  │ 6DoF Head Pose Est.     │  │
                          │  │ Expression Coefficients │  │
                          │  └─────────────────────────┘  │
                          └───────────┬───────────────────┘
                                      │
                          ┌───────────▼───────────────────┐
                          │   Talking Head Engine          │
                          │  ├─ Head pose from audio      │
                          │  ├─ Expression from speech    │
                          │  └─ Natural head motion       │
                          └───────────┬───────────────────┘
                                      │
                          ┌───────────▼───────────────────┐
                          │   Lip Sync (Wav2Lip)           │
                          │  ├─ Phoneme detection         │
                          │  ├─ Lip shape prediction      │
                          │  ├─ Wav2Lip inference         │
                          │  └─ Temporal smoothness       │
                          └───────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
     ┌────────▼────────┐   ┌─────────▼─────────┐   ┌─────────▼────────┐
     │  Eye & Gaze      │   │  Gesture Engine    │   │  Physics Engine  │
     │  Engine          │   │  ────────────────  │   │  ─────────────  │
     │  ├─ Saccades     │   │  Hand gestures    │   │  Momentum/inert│
     │  ├─ Blinks       │   │  Shoulder/head    │   │  Spring dynamics│
     │  ├─ Gaze track   │   │  Posture          │   │  Frame skipping │
     │  └─ Pupil render │   │  Intensity config │   │  Async queues   │
     └────────┬────────┘   └─────────┬─────────┘   └─────────┬────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
                          ┌───────────▼───────────────────┐
                          │   Composite & Render           │
                          │  Face Swap + All Overlays     │
                          │  + Background + Enhancement   │
                          └───────────┬───────────────────┘
                                      │
                          ┌───────────▼───────────────────┐
                          │   Output                      │
                          │  ┌────────┐ ┌───────┐ ┌────┐ │
                          │  │Streamlit│ │FastAPI│ │Virt│ │
                          │  │UI :8080│ │:8081  │ │Cam │ │
                          │  └────────┘ └───────┘ └────┘ │
                          └──────────────────────────────┘
```

## 🔧 Engine Documentation

### 1. 3D Face Engine (`core/face_3d_engine.py`)
- **MediaPipe FaceMesh**: 468 landmark detection
- **Delaunay Triangulation**: Converts landmarks to 3D mesh surface
- **6DoF Head Pose**: Pitch, yaw, roll, x, y, z estimation using solvePnP
- **Expression Coefficients**: Extract and apply expression blendshapes
- **Usage**: `face_3d = get_face_3d_engine(); mesh = face_3d.process_frame(image)`

### 2. Audio-Driven Talking Head (`core/talking_head.py`)
- **MFCC Extraction**: 13 MFCC coefficients from audio chunks
- **Pitch Detection**: Fundamental frequency (F0) estimation
- **Energy Envelope**: RMS energy for speech emphasis
- **Head Pose Prediction**: Audio features → head motion mapping
- **Usage**: `talking_head = get_talking_head(); seq = talking_head.process_audio(audio_data, face_img)`

### 3. Wav2Lip Lip Sync (`core/lip_sync.py`)
- **Phoneme Detection**: Maps audio to phoneme classes
- **Lip Shape Prediction**: Audio features → mouth shape parameters
- **Wav2Lip Integration**: Model download + inference on face region
- **Temporal Smoothing**: EMA filter for natural transitions
- **Usage**: `lip_sync = create_lip_sync(); frame = lip_sync.sync_frame(face_img, audio_chunk)`

### 4. Eye & Gaze Engine (`core/eye_engine.py`)
- **Saccade Simulation**: Rapid eye jumps every 200-300ms
- **Micro-Saccades**: Tiny movements during fixation periods
- **Blink Pattern**: Every 2-4 seconds, 100-400ms duration
- **Gaze Tracking**: Smooth pursuit with configurable targets
- **Usage**: `eye_engine = get_eye_engine(); state = eye_engine.update()`

### 5. Gesture Engine (`core/gesture_engine.py`)
- **Speech-Rhythm Gestures**: Hand patterns matching audio energy
- **Micro-Movements**: Shoulder/head shifts during speech
- **Posture Variations**: Natural position changes
- **Intensity Control**: Configurable 0.0-1.0
- **Usage**: `gesture = get_gesture_engine(); params = gesture.process_gestures(energy)`

### 6. Real-Time Pipeline (`core/pipeline.py`)
- **Async Queue System**: Multi-stage async pipeline with per-engine queues
- **Frame Skipping**: Skip every Nth frame for CPU optimization
- **Cached Inference**: Cache Wav2Lip results for repeated phonemes
- **Resolution Management**: Downscale for detection, upscale for output
- **Usage**: `pipeline = get_realtime_pipeline(); pipeline.start()`

## 💻 Professional UI

The Streamlit UI (`app.py`) runs on port **8080** and includes:

| Tab | Features |
|-----|----------|
| 🎯 **Avatar Studio** | Source photo upload, real-time preview, recording export |
| 📱 **Mobile** | QR code generator for IP Webcam, auto-detect, camera source selector |
| 👤 **Characters** | Gallery, creation, management with face data |
| 💬 **Chat** | LLM character conversation with message history |
| 🎬 **Engines** | Live status for all 6 engines, per-engine stage timing |
| ⚙️ **Settings** | All engine toggles, sliders, quality controls, HeyGen Mode preset |

### Heygen Mode
One-click "HeyGen Mode" preset enables all engines at maximum quality:
- ✅ 3D Face Engine | ✅ Talking Head | ✅ Wav2Lip | ✅ Eye Gaze | ✅ Gestures
- ✅ Parallax Background | ✅ Physics | ✅ High Quality Enhancement

## 🔌 API v2

The FastAPI backend (`api.py`) runs on port **8081** with auto-generated docs at `/docs`.

### New v2.0.0 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| **POST** | **`/generate/talking-head`** | Generate talking head video from audio + face image |
| **POST** | **`/animate/face`** | Animate face with expression coefficients + head pose |
| **WS** | **`/ws/realtime`** | Real-time streaming with head pose + eye state |
| **POST** | **`/config/render`** | Configure any engine's render parameters |

### Legacy Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info + credits |
| GET | `/status` | Full system status with all engine FPS |
| POST | `/swap` | Swap face on uploaded image |
| POST | `/chat` | Send message to character LLM |
| GET/POST/DELETE | `/characters` | Character CRUD |
| POST | `/characters/{name}/activate` | Activate character |
| POST | `/physics/config` | Update physics parameters |
| GET | `/physics/status` | Current physics state |
| POST | `/camera/source` | Switch camera source |
| GET | `/camera/status` | Current camera info |
| WS | `/ws/chat` | Streaming chat |
| WS | `/ws/video` | Real-time video |

### Talking Head API Example

```bash
curl -X POST http://localhost:8081/generate/talking-head \
  -H "Content-Type: application/json" \
  -d '{
    "audio_b64": "BASE64_ENCODED_WAV_AUDIO",
    "face_b64": "BASE64_ENCODED_FACE_IMAGE",
    "fps": 20
  }'
```

### Render Config Example

```bash
curl -X POST http://localhost:8081/config/render \
  -H "Content-Type: application/json" \
  -d '{"engine": "eye", "config": {"blink_interval_min": 1.5, "blink_interval_max": 4.0}}'
```

## 🪟 Build Windows EXE

Build a standalone Windows executable with PyInstaller:

```bash
# On Windows
pip install pyinstaller
python build_exe.py

# Output: dist/DeepFaceReal.exe + DeepFaceReal_API.exe
```

The EXE bundles:
- All core modules (face_3d_engine, talking_head, lip_sync, eye_engine, gesture_engine, pipeline)
- InsightFace models (buffalo_l, inswapper_128)
- MediaPipe models
- Wav2Lip models (if downloaded)
- OpenCV, NumPy, Pillow, Streamlit, FastAPI
- Launcher batch file for easy startup

## 📱 WhatsApp Integration

### Mobile (Android)
1. Install **IP Webcam** from Google Play Store
2. Open app → tap **Start Server**
3. Note IP address (e.g., `192.168.1.100:8080`)
4. Open Streamlit UI → **📱 Mobile** tab → enter IP or scan QR code

### Desktop
1. `sudo apt install v4l2loopback-dkms`
2. `sudo modprobe v4l2loopback devices=1 video_nr=10`
3. Start pipeline → virtual camera appears as device
4. Select "DeepFakeCam" in WhatsApp Desktop / Zoom / Meet

## 🛠️ Project Structure

```
DeepFaceReal-Physics/
├── app.py                    # Streamlit UI v2 (Port 8080)
├── api.py                    # FastAPI v2 (Port 8081)
├── build_exe.py              # Windows EXE builder
├── README.md                 # This file
├── start.sh                  # Launch both services
├── core/
│   ├── __init__.py           # Module exports
│   ├── face_3d_engine.py     # 3D Face Reconstruction + Pose
│   ├── talking_head.py       # Audio-Driven Talking Head
│   ├── lip_sync.py           # Wav2Lip Lip Sync
│   ├── eye_engine.py         # Natural Eye & Gaze
│   ├── gesture_engine.py     # Conversational Gestures
│   ├── pipeline.py           # Real-Time Pipeline
│   ├── face_swapper.py       # InsightFace swap (v1 enhanced)
│   ├── physics_engine.py     # MediaPipe Holistic + physics (v1 preserved)
│   ├── background_engine.py   # Parallax background (v1 preserved)
│   ├── webcam_pipeline.py    # Camera capture (v1 preserved)
│   ├── character_manager.py  # Character profiles (v1 preserved)
│   └── llm_character.py      # OpenRouter LLM (v1 preserved)
├── models/                   # Model downloads
├── profiles/                 # Character profiles
└── static/                   # Static assets
```

## 📊 Performance

| Engine | Target FPS | CPU Cores | Resolution |
|--------|-----------|-----------|------------|
| 3D Face | 30 FPS | 1 | 640×480 |
| Talking Head | 30 FPS | 1 | Audio only |
| Lip Sync | 20 FPS | 1 | Face region |
| Eye Gaze | 60 FPS | 0.5 | Overlay |
| Gestures | 30 FPS | 0.5 | Overlay |
| Pipeline Total | **≥15-20 FPS** | **4 cores** | **640×480** |

> All engines optimized for CPU with frame skipping (process every 2nd-3rd frame for heavy stages), cached inference for repeated phonemes, and async pipeline queues.

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Credits

<div align="center">
<br>
<h2>🎭 DeepFaceReal-Physics v2.0.0</h2>
<h3>Hyper-Realistic AI Avatar with HeyGen-Level Realism</h3>
<br>
<p><strong>Powered By <span style="color:#ff6b6b;">DeathLegionTeamLK</span></strong></p>
<br>
<p>Built with:</p>
<p>
<a href="https://github.com/deepinsight/insightface">InsightFace</a> •
<a href="https://google.github.io/mediapipe/">MediaPipe</a> •
<a href="https://github.com/iperov/DeepFaceLab">Wav2Lip</a> •
<a href="https://onnxruntime.ai/">ONNX Runtime</a> •
<a href="https://openrouter.ai/">OpenRouter</a> •
<a href="https://streamlit.io/">Streamlit</a> •
<a href="https://fastapi.tiangolo.com/">FastAPI</a> •
<a href="https://scipy.org/">SciPy</a> •
<a href="https://scikit-image.org/">scikit-image</a>
</p>
<br>
<p>
<a href="https://github.com/deathlegionteamlk/DeepFaceReal-Physics/stargazers">⭐ Star us on GitHub</a>
</p>
<br>
<p><em>Inspired by SadTalker, Ditto, and the HeyGen platform.</em></p>
<br>
</div>

---

<div align="center">
<small>DeepFaceReal-Physics v2.0.0 | Powered By DeathLegionTeamLK | MIT License</small>
</div>