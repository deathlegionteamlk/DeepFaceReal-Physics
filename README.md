<div align="center">

![DeepFaceReal-Physics Banner](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20&height=200&section=header&text=DeepFaceReal-Physics&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Hyper-Realistic%20AI%20Avatar%20%E2%80%94%20HeyGen-Level%20Realism%2C%20Zero%20Monthly%20Fee&descAlignY=62&descAlign=50)

[![GitHub Stars](https://img.shields.io/github/stars/deathlegionteamlk/DeepFaceReal-Physics?style=social)](https://github.com/deathlegionteamlk/DeepFaceReal-Physics)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![GitHub Release](https://img.shields.io/github/v/release/deathlegionteamlk/DeepFaceReal-Physics)](https://github.com/deathlegionteamlk/DeepFaceReal-Physics/releases)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/deathlegionteamlk/DeepFaceReal-Physics/graphs/commit-activity)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/Built%20with-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

<br>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=FF6B6B&center=true&vCenter=true&multiline=true&width=700&height=80&lines=3D+Face+%E2%80%A2+Lip+Sync+%E2%80%A2+Eye+Gaze+%E2%80%A2+Gestures+%E2%80%A2+Physics;CPU-only+%E2%80%A2+Real-time+%E2%80%A2+Open+Source+%E2%80%A2+Free+Forever" alt="Typing SVG" />

<br>

<p>
<a href="#-features"><strong>Features</strong></a> •
<a href="#-heygen-vs-sadtalker-vs-deepfacereal"><strong>Comparison</strong></a> •
<a href="#-quick-start"><strong>Quick Start</strong></a> •
<a href="#-architecture"><strong>Architecture</strong></a> •
<a href="#-engine-docs"><strong>Engine Docs</strong></a> •
<a href="#-api-v2"><strong>API v2</strong></a> •
<a href="#-windows-exe"><strong>Windows EXE</strong></a> •
<a href="#-credits"><strong>Credits</strong></a>
</p>

<br>

[🌟 Star on GitHub](https://github.com/deathlegionteamlk/DeepFaceReal-Physics) • [🐛 Report Bug](https://github.com/deathlegionteamlk/DeepFaceReal-Physics/issues) • [📖 Read Docs](https://github.com/deathlegionteamlk/DeepFaceReal-Physics/blob/main/README.md)

</div>

---

## 🎬 What Is This?

HeyGen costs $24–240/month and runs in the cloud. SadTalker needs a GPU and still only hits 10 FPS. DeepFaceReal-Physics runs on your CPU, is free, and ships everything: 3D face reconstruction, audio-driven head motion, Wav2Lip lip sync, natural eye gaze, conversational gestures, body physics — the full stack.

No subscription. No GPU required. No waiting on cloud queues.

---

## ✨ Features

<div align="center">

![Features Animation](https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=2,3,30&height=3&section=header)

</div>

| Engine | What It Does |
|--------|-------------|
| 🎯 **3D Face** | 468 MediaPipe landmarks, Delaunay triangulation, 6DoF head pose (pitch/yaw/roll/xyz), expression blendshapes |
| 🗣️ **Talking Head** | MFCC/pitch/energy extraction, audio-to-head-pose mapping, natural nodding/tilting patterns |
| 👄 **Wav2Lip** | Phoneme detection, lip shape prediction, temporal smoothing, real-time audio buffering |
| 👁️ **Eye & Gaze** | Saccades every 200–300ms, natural blinks every 2–4s, gaze target tracking, pupil rendering |
| 👐 **Gestures** | Speech-rhythm hand movement, shoulder/head micro-shifts, posture variation, 0.0–1.0 intensity knob |
| 🔄 **Pipeline** | Async multi-stage queue, frame skipping, cached inference — all CPU-optimized |
| 🧠 **Body Physics** | MediaPipe Holistic (543 landmarks), momentum/inertia, spring dynamics |
| 🖼️ **Parallax BG** | 3-layer depth parallax driven by head position, depth blur |
| 📱 **Mobile Camera** | IP Webcam integration for Android phone as webcam |
| 💬 **Characters** | OpenRouter LLM with personality system prompts |
| 🖥️ **UI** | Streamlit v2 dashboard, HeyGen Mode preset, recording export |
| 🔌 **API** | FastAPI v2 with REST + WebSocket endpoints |
| 🪟 **Windows EXE** | PyInstaller standalone build |

---

## 📊 HeyGen vs SadTalker vs DeepFaceReal

<div align="center">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=16&duration=2000&pause=800&color=00D4AA&center=true&vCenter=true&width=500&lines=Spoiler%3A+free+wins." alt="Spoiler" />
</div>

| Capability | **HeyGen** ($24–240/mo) | **SadTalker** | **DeepFaceReal-Physics** |
|------------|:---:|:---:|:---:|
| 3D Face Reconstruction | ✅ | ❌ | ✅ |
| Audio-Driven Head Motion | ✅ | ✅ | ✅ |
| Wav2Lip Lip Sync | ✅ | ❌ | ✅ |
| Natural Eye Gaze | ✅ | ❌ | ✅ |
| Conversational Gestures | ✅ | ❌ | ✅ |
| Real-Time ≥15 FPS | ❌ Cloud | ⚠️ 10 FPS GPU | ✅ **15–20 FPS CPU** |
| Face Swap | ❌ | ❌ | ✅ |
| LLM Character AI | ⚠️ Limited | ❌ | ✅ |
| Self-Hosted | ❌ | ✅ | ✅ |
| Open Source | ❌ | ✅ | ✅ |
| **Price** | **$24–240/month** | Free | **Free** |
| WhatsApp Integration | ❌ | ❌ | ✅ |
| Windows EXE | N/A | Manual | ✅ |
| API + WebSocket | ✅ | ❌ | ✅ |
| GPU Required | ✅ | ✅ | ❌ **CPU only** |

---

## 🚀 Quick Start

<div align="center">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=15&duration=1800&pause=600&color=FFD700&center=true&vCenter=true&width=400&lines=Clone+%E2%86%92+Install+%E2%86%92+Run.+That%27s+it." alt="Quick start hint" />
</div>

**Prerequisites:** Python 3.10+, 4GB RAM (8GB recommended), webcam

### Install & Run

```bash
# Clone
git clone https://github.com/deathlegionteamlk/DeepFaceReal-Physics.git
cd DeepFaceReal-Physics

# Install
pip install -r requirements.txt

# Start the UI (port 8080)
streamlit run app.py --server.port 8080

# In a second terminal — start the API (port 8081)
python api.py
```

### Docker

```bash
docker build -t deepfacereal-physics .
docker run -p 8080:8080 -p 8081:8081 deepfacereal-physics
```

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────┐
                    │        Input Sources          │
                    │  ┌──────┐ ┌────────┐ ┌─────┐ │
                    │  │ USB  │ │   IP   │ │Audio│ │
                    │  │ Cam  │ │ Webcam │ │File │ │
                    │  └──┬───┘ └───┬────┘ └──┬──┘ │
                    └─────┼─────────┼──────────┼───┘
                          │         │          │
                    ┌─────▼─────────▼──────────▼───┐
                    │    Audio Feature Extraction   │
                    │   (MFCC · Pitch · Energy · F0)│
                    └──────────────┬────────────────┘
                                   │
                    ┌──────────────▼────────────────┐
                    │         3D Face Engine         │
                    │  MediaPipe 468 landmarks       │
                    │  Delaunay Triangulation        │
                    │  6DoF Head Pose (solvePnP)     │
                    │  Expression Blendshapes        │
                    └──────────────┬────────────────┘
                                   │
                    ┌──────────────▼────────────────┐
                    │       Talking Head Engine      │
                    │  Audio → head pose             │
                    │  Audio → expression            │
                    │  Natural motion patterns       │
                    └──────────────┬────────────────┘
                                   │
                    ┌──────────────▼────────────────┐
                    │       Lip Sync (Wav2Lip)       │
                    │  Phoneme detection             │
                    │  Lip shape prediction          │
                    │  Wav2Lip inference             │
                    │  Temporal smoothing            │
                    └──────────────┬────────────────┘
                                   │
          ┌────────────────────────┼─────────────────────────┐
          │                        │                         │
 ┌────────▼────────┐    ┌──────────▼──────────┐   ┌─────────▼───────┐
 │  Eye & Gaze     │    │   Gesture Engine     │   │ Physics Engine  │
 │  Saccades       │    │   Hand gestures      │   │ Momentum        │
 │  Blinks         │    │   Shoulder/head      │   │ Spring dynamics │
 │  Gaze tracking  │    │   Posture shifts     │   │ Frame skipping  │
 │  Pupil render   │    │   Intensity config   │   │ Async queues    │
 └────────┬────────┘    └──────────┬──────────┘   └─────────┬───────┘
          │                        │                         │
          └────────────────────────┼─────────────────────────┘
                                   │
                    ┌──────────────▼────────────────┐
                    │      Composite & Render        │
                    │  Face swap · overlays          │
                    │  Background · enhancement      │
                    └──────────────┬────────────────┘
                                   │
                    ┌──────────────▼────────────────┐
                    │            Output              │
                    │  Streamlit :8080               │
                    │  FastAPI   :8081               │
                    │  Virtual Camera                │
                    └───────────────────────────────┘
```

---

## 🔧 Engine Docs

### 1. 3D Face Engine — `core/face_3d_engine.py`

468 MediaPipe landmarks → Delaunay triangulation → 3D mesh. 6DoF head pose via `solvePnP`. Expression blendshape extraction.

```python
face_3d = get_face_3d_engine()
mesh = face_3d.process_frame(image)
```

### 2. Talking Head — `core/talking_head.py`

13 MFCC coefficients, F0 pitch, RMS energy → head pose prediction. Drives speech-synchronized nodding and tilt.

```python
talking_head = get_talking_head()
seq = talking_head.process_audio(audio_data, face_img)
```

### 3. Wav2Lip Lip Sync — `core/lip_sync.py`

Phoneme detection → lip shape parameters → Wav2Lip inference on face region. EMA filter keeps transitions smooth.

```python
lip_sync = create_lip_sync()
frame = lip_sync.sync_frame(face_img, audio_chunk)
```

### 4. Eye & Gaze Engine — `core/eye_engine.py`

Saccades every 200–300ms, micro-saccades during fixation, blinks every 2–4s (100–400ms duration). Configurable gaze targets.

```python
eye_engine = get_eye_engine()
state = eye_engine.update()
```

### 5. Gesture Engine — `core/gesture_engine.py`

Hand patterns keyed to audio energy, shoulder/head micro-movements, posture variation. Intensity from 0.0 to 1.0.

```python
gesture = get_gesture_engine()
params = gesture.process_gestures(energy)
```

### 6. Real-Time Pipeline — `core/pipeline.py`

Async queue per stage. Frame skipping for CPU relief. Cached Wav2Lip results for repeated phonemes. Resolution management (downscale detect, upscale output).

```python
pipeline = get_realtime_pipeline()
pipeline.start()
```

---

## 💻 Professional UI

The Streamlit UI (`app.py`) runs on port **8080**.

| Tab | What's Inside |
|-----|---------------|
| 🎯 **Avatar Studio** | Source photo upload, real-time preview, recording export |
| 📱 **Mobile** | QR code for IP Webcam, auto-detect, camera source picker |
| 👤 **Characters** | Gallery, creation, management with face data |
| 💬 **Chat** | LLM character conversation with message history |
| 🎬 **Engines** | Live status for all 6 engines, per-stage timing |
| ⚙️ **Settings** | Engine toggles, sliders, quality controls, HeyGen Mode preset |

### HeyGen Mode

One click. Turns everything on at max quality:

✅ 3D Face · ✅ Talking Head · ✅ Wav2Lip · ✅ Eye Gaze · ✅ Gestures · ✅ Parallax BG · ✅ Physics · ✅ High Quality Enhancement

---

## 🔌 API v2

FastAPI (`api.py`) on port **8081**. Auto-generated docs at `/docs`.

### v2 Endpoints

| Method | Path | What It Does |
|--------|------|-------------|
| POST | `/generate/talking-head` | Generate talking head video from audio + face image |
| POST | `/animate/face` | Animate face with expression coefficients + head pose |
| WS | `/ws/realtime` | Real-time streaming with head pose + eye state |
| POST | `/config/render` | Configure any engine's render parameters |

### Legacy Endpoints

| Method | Path | |
|--------|------|-|
| GET | `/` | API info |
| GET | `/status` | System status with per-engine FPS |
| POST | `/swap` | Face swap on uploaded image |
| POST | `/chat` | Send message to character LLM |
| GET/POST/DELETE | `/characters` | Character CRUD |
| POST | `/characters/{name}/activate` | Activate character |
| POST/GET | `/physics/config`, `/physics/status` | Physics control |
| POST/GET | `/camera/source`, `/camera/status` | Camera control |
| WS | `/ws/chat`, `/ws/video` | Streaming chat + video |

### Talking Head — Example

```bash
curl -X POST http://localhost:8081/generate/talking-head \
  -H "Content-Type: application/json" \
  -d '{
    "audio_b64": "BASE64_ENCODED_WAV_AUDIO",
    "face_b64": "BASE64_ENCODED_FACE_IMAGE",
    "fps": 20
  }'
```

### Render Config — Example

```bash
curl -X POST http://localhost:8081/config/render \
  -H "Content-Type: application/json" \
  -d '{"engine": "eye", "config": {"blink_interval_min": 1.5, "blink_interval_max": 4.0}}'
```

---

## 🪟 Windows EXE

```bash
# On Windows
pip install pyinstaller
python build_exe.py

# Output: dist/DeepFaceReal.exe + DeepFaceReal_API.exe
```

The build bundles all core modules, InsightFace models (buffalo_l, inswapper_128), MediaPipe models, Wav2Lip models, OpenCV/NumPy/Pillow/Streamlit/FastAPI, and a launcher batch file.

---

## 📱 Mobile Integration

### Android (IP Webcam)

1. Install **IP Webcam** from Google Play
2. Tap **Start Server**
3. Note the IP (e.g. `192.168.1.100:8080`)
4. In Streamlit → **📱 Mobile** tab → enter IP or scan QR code

### Desktop Virtual Camera

```bash
sudo apt install v4l2loopback-dkms
sudo modprobe v4l2loopback devices=1 video_nr=10
```

Start the pipeline → virtual camera appears as a device → select "DeepFakeCam" in WhatsApp Desktop, Zoom, or Meet.

---

## 🛠️ Project Structure

```
DeepFaceReal-Physics/
├── app.py                    # Streamlit UI v2 (port 8080)
├── api.py                    # FastAPI v2 (port 8081)
├── build_exe.py              # Windows EXE builder
├── start.sh                  # Launch both services
├── core/
│   ├── face_3d_engine.py     # 3D face reconstruction + pose
│   ├── talking_head.py       # Audio-driven talking head
│   ├── lip_sync.py           # Wav2Lip lip sync
│   ├── eye_engine.py         # Eye & gaze
│   ├── gesture_engine.py     # Conversational gestures
│   ├── pipeline.py           # Real-time async pipeline
│   ├── face_swapper.py       # InsightFace swap
│   ├── physics_engine.py     # MediaPipe Holistic + physics
│   ├── background_engine.py  # Parallax background
│   ├── webcam_pipeline.py    # Camera capture
│   ├── character_manager.py  # Character profiles
│   └── llm_character.py      # OpenRouter LLM
├── models/                   # Downloaded models
├── profiles/                 # Character profiles
└── static/                   # Static assets
```

---

## 📊 Performance

| Engine | Target FPS | CPU Cores | Resolution |
|--------|-----------|-----------|------------|
| 3D Face | 30 FPS | 1 | 640×480 |
| Talking Head | 30 FPS | 1 | Audio only |
| Lip Sync | 20 FPS | 1 | Face region |
| Eye Gaze | 60 FPS | 0.5 | Overlay |
| Gestures | 30 FPS | 0.5 | Overlay |
| **Pipeline Total** | **≥15–20 FPS** | **4 cores** | **640×480** |

Every heavy stage processes every 2nd–3rd frame. Repeated phonemes hit the Wav2Lip cache. Async queues keep everything non-blocking.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit (`git commit -m 'Add your feature'`)
4. Push (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Credits

<div align="center">

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20&height=120&section=footer&text=Powered%20By%20DeathLegionTeamLK&fontSize=28&fontColor=ffffff&animation=fadeIn)

Built on:

[InsightFace](https://github.com/deepinsight/insightface) • [MediaPipe](https://google.github.io/mediapipe/) • [Wav2Lip](https://github.com/iperov/DeepFaceLab) • [ONNX Runtime](https://onnxruntime.ai/) • [OpenRouter](https://openrouter.ai/) • [Streamlit](https://streamlit.io/) • [FastAPI](https://fastapi.tiangolo.com/) • [SciPy](https://scipy.org/) • [scikit-image](https://scikit-image.org/)

Inspired by SadTalker, Ditto, and HeyGen.

[⭐ Star on GitHub](https://github.com/deathlegionteamlk/DeepFaceReal-Physics/stargazers)

<br>
<small>DeepFaceReal-Physics v2.0.0 · MIT License · DeathLegionTeamLK</small>

</div>
