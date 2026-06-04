[![GitHub Stars](https://img.shields.io/github/stars/deathlegionteamlk/DeepFaceReal-Physics?style=social)](https://github.com/deathlegionteamlk/DeepFaceReal-Physics)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![GitHub Release](https://img.shields.io/github/v/release/deathlegionteamlk/DeepFaceReal-Physics)](https://github.com/deathlegionteamlk/DeepFaceReal-Physics/releases)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/deathlegionteamlk/DeepFaceReal-Physics/graphs/commit-activity)

<meta name="description" content="DeepFaceReal Physics - State-of-the-art real-time deepfake with full body tracking (MediaPipe Holistic), physics simulation (inertia, springs, momentum), 3-layer parallax background, WhatsApp Mobile/Desktop integration, and LLM character AI responses. CPU-optimized with no GPU required.">
<meta name="keywords" content="deepfake, face swap, real-time, mediapipe holistic, full body tracking, physics simulation, parallax background, ip webcam, whatsapp integration, character AI, openrouter, insightface, cpu optimized, open source">

<div align="center">
  <h1>🎭 DeepFaceReal Physics</h1>
  <p><strong>State-of-the-art Real-Time Deepfake with Full Body Tracking, Physics Simulation, and WhatsApp Mobile/Desktop Integration</strong></p>
  <p><em>Powered By <strong>DeathLegionTeamLK</strong></em></p>
  <br>
  <p>
    <a href="#features">Features</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#whatsapp-integration">WhatsApp Mobile + Desktop</a> •
    <a href="#physics-system">Physics System</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#api-documentation">API</a> •
    <a href="#credits">Credits</a>
  </p>
  <br>
</div>

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 🔄 **Real-Time Face Swap** | InsightFace inswapper_128 + buffalo_l for high-quality face detection and swapping |
| 🧠 **Full Body Tracking** | MediaPipe Holistic: 33 pose + 21×2 hands + 468 face mesh = **543 landmarks total** |
| 🎯 **Physics Simulation** | Momentum/inertia, spring dynamics, hand physics, micro-expression & blink simulation |
| 🖼️ **3-Layer Parallax Background** | Far/mid/near depth planes with head-position-driven shift and depth-based blur |
| 📱 **WhatsApp Mobile Support** | IP Webcam integration — use your phone camera for face swapping on calls |
| 💬 **LLM Character AI** | OpenRouter-powered character responses with personality system prompts |
| 🎥 **Virtual Camera Output** | v4l2loopback integration for use in WhatsApp Desktop, Zoom, Google Meet |
| 🖥️ **Web Dashboard** | Streamlit UI with live preview, character gallery, physics controls, and chat |
| 🔌 **REST API + WebSocket** | FastAPI backend with full control via REST endpoints and real-time video/chat streaming |
| 🎨 **Face Enhancement** | Poisson seamless cloning, bilateral filtering, sharpening, and CLAHE equalization |
| 🔬 **CPU-Optimized** | Frame skipping, resolution management, ONNX Runtime — no GPU required |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 4GB+ RAM (8GB+ recommended)
- Webcam (USB or Android phone via IP Webcam)
- Linux with v4l2loopback (for virtual camera / WhatsApp Desktop)

### Installation

```bash
# Clone the repository
git clone https://github.com/deathlegionteamlk/DeepFaceReal-Physics.git
cd DeepFaceReal-Physics

# Install dependencies
pip install -r requirements.txt

# Download models (automatically on first run)
# InsightFace buffalo_l + inswapper_128.onnx

# Run the web dashboard
python app.py
```

### Run Services

```bash
# Streamlit Web Dashboard (port 8080)
python app.py

# FastAPI Backend (port 8081)
python api.py

# Start both with one command
bash start.sh
```

---

## 📱 WhatsApp Integration

DeepFaceReal Physics supports **both WhatsApp Mobile and WhatsApp Desktop** for face swapping in video calls.

### 📲 WhatsApp Mobile (IP Webcam)

Use your Android phone as the camera source for face swapping:

1. Install **IP Webcam** from Google Play Store
2. Open the app → scroll to **Start server**
3. Note the IP address (e.g., `http://192.168.1.100:8080`)
4. In the Streamlit UI, go to **📱 Mobile Camera** tab
5. Enter the phone IP or scan the QR code
6. The face swap pipeline receives your phone camera feed

### 🖥️ WhatsApp Desktop (Virtual Camera)

Use the face swap output as a virtual camera in WhatsApp Desktop:

1. Install and load the virtual camera module:
   ```bash
   sudo apt install v4l2loopback-dkms v4l2loopback-utils
   sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="DeepFakeCam" exclusive_caps=1
   ```
2. Start the face swap pipeline
3. Open WhatsApp Desktop → start a video call
4. Click **⋮** → **Settings** → **Camera** → Select **DeepFakeCam**
5. Your swapped face appears in real-time

📖 See [WHATSAPP_SETUP.md](WHATSAPP_SETUP.md) for detailed setup instructions.

---

## 🔬 Physics System

### Components

| Component | Description | Key Parameters |
|-----------|-------------|----------------|
| **Momentum & Inertia** | Tracks landmark velocity per frame, applies momentum to overlay elements | `mass` (0.5–5.0), `damping` (0.7–0.95) |
| **Spring Physics** | Head tilt drives spring-based overlay motion (hair, accessories) | `spring_stiffness` (0.1–0.5), `damping` |
| **Hand Physics** | Natural arm/hand position overlay driven by detected landmarks | Lerp smoothing with configurable alpha |
| **Face Physics** | Micro-expression simulation, EAR-based blink detection, head bob | `intensity` (0.0–1.0) |

### Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Webcam    │────▶│  MediaPipe       │────▶│  Physics     │
│   Frame     │     │  Holistic        │     │  Engine      │
└─────────────┘     └──────────────────┘     └──────────────┘
                          │                        │
                          ▼                        ▼
                    ┌──────────────┐     ┌──────────────────┐
                    │  543         │     │  Momentum/Spring │
                    │  Landmarks   │     │  Hand/Face Phys  │
                    └──────────────┘     └──────────────────┘
                                                    │
                          ┌─────────┐              │
                          │ 3-Layer │◀─────────────┘
                          │ Parallax│
                          │ BG      │
                          └─────────┘
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DeepFaceReal Physics                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌───────────────┐  │
│  │ Webcam   │──▶│ Insight   │──▶│ Physics  │──▶│ Virtual       │  │
│  │ /IP Cam  │   │ Face Swap │   │ + BG     │   │ Camera Output │  │
│  └──────────┘   └───────────┘   └──────────┘   └───────────────┘  │
│       │               │               │                            │
│       ▼               ▼               ▼                            │
│  ┌──────────┐   ┌───────────┐   ┌──────────────┐                  │
│  │ MJPEG    │   │ MediaPipe │   │ Holistic     │                  │
│  │ Stream   │   │ Holistic  │   │ Tracker      │                  │
│  └──────────┘   └───────────┘   └──────────────┘                  │
│                                                                     │
│  ┌──────────┐   ┌───────────┐   ┌──────────────┐                  │
│  │ OpenRouter│──▶│ Character │──▶│ Streamlit    │                  │
│  │ LLM API  │   │ Manager   │   │ Web UI       │                  │
│  └──────────┘   └───────────┘   └──────────────┘                  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  FastAPI Backend (REST + WebSocket)                         │  │
│  │  - POST /swap  - POST /chat  - POST /physics/config        │  │
│  │  - GET /status - GET /physics/status - POST /camera/source │  │
│  │  - WebSocket /ws/video - WebSocket /ws/chat                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔌 API Documentation

The FastAPI backend runs on port 8081 with auto-generated docs at `/docs`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info + credits |
| GET | `/status` | System status |
| POST | `/swap` | Swap face on uploaded image |
| POST | `/chat` | Send message to character LLM |
| GET/POST/DELETE | `/characters` | Character CRUD |
| POST | `/characters/{name}/activate` | Activate character |
| **POST** | **`/physics/config`** | Update physics parameters |
| **GET** | **`/physics/status`** | Current physics state + FPS |
| **POST** | **`/camera/source`** | Switch camera source (usb/ip_webcam) |
| **GET** | **`/camera/status`** | Current camera info |
| WS | `/ws/chat` | Streaming chat via WebSocket |
| WS | `/ws/video` | Real-time video via WebSocket |

### Physics Configuration Example

```bash
curl -X POST http://localhost:8081/physics/config \
  -H "Content-Type: application/json" \
  -d '{"mass": 1.5, "damping": 0.9, "spring_stiffness": 0.3, "intensity": 0.7}'
```

---

## 🛠️ Project Structure

```
real_time_deepfake_0851/
├── app.py                          # Streamlit Web Dashboard (port 8080)
├── api.py                          # FastAPI Backend (port 8081)
├── core/
│   ├── __init__.py                 # Module exports
│   ├── face_swapper.py             # InsightFace swap + physics/background integration
│   ├── physics_engine.py           # MediaPipe Holistic tracker + physics simulation
│   ├── background_engine.py        # 3-layer parallax background system
│   ├── webcam_pipeline.py          # Camera capture + virtual camera + IP Webcam
│   ├── character_manager.py        # Character profile CRUD
│   ├── llm_character.py           # OpenRouter LLM character responses
│   └── lip_sync.py                 # Amplitude-based lip sync
├── models/                         # Download directory for models
├── profiles/                       # Character profile JSON files
├── WHATSAPP_SETUP.md               # WhatsApp integration guide
├── CHARACTERS.md                   # Character management guide
├── README.md                       # This file
└── start.sh                        # Launch both services
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Credits

<div align="center">
  <br>
  <h3>🎭 DeepFaceReal Physics</h3>
  <p><strong>Powered By <span style="color: #ff6b6b;">DeathLegionTeamLK</span></strong></p>
  <br>
  <p>Built with:</p>
  <p>
    <a href="https://github.com/deepinsight/insightface">InsightFace</a> •
    <a href="https://google.github.io/mediapipe/">MediaPipe</a> •
    <a href="https://onnxruntime.ai/">ONNX Runtime</a> •
    <a href="https://openrouter.ai/">OpenRouter</a> •
    <a href="https://streamlit.io/">Streamlit</a> •
    <a href="https://fastapi.tiangolo.com/">FastAPI</a>
  </p>
  <br>
</div>