# WhatsApp Desktop Integration Guide

Use the Real-Time Deepfake System as a virtual camera in WhatsApp Desktop video calls.

## Prerequisites

- Linux system with `v4l2loopback` kernel module
- Real-Time Deepfake System installed and running
- WhatsApp Desktop app installed

## Step 1: Install Virtual Camera

```bash
# Install v4l2loopback
sudo apt update
sudo apt install v4l2loopback-dkms v4l2loopback-utils

# Load the kernel module with a virtual camera device
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="DeepFakeCam" exclusive_caps=1
```

**Persistence**: To load automatically on boot:
```bash
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo "options v4l2loopback devices=1 video_nr=10 card_label=DeepFakeCam exclusive_caps=1" | sudo tee /etc/modprobe.d/v4l2loopback.conf
```

## Step 2: Verify Virtual Camera

```bash
# List video devices
v4l2-ctl --list-devices

# You should see:
# DeepFakeCam (platform:v4l2loopback):
#     /dev/video10
```

## Step 3: Start the Deepfake Pipeline

```bash
# 1. Start the API backend
cd /app/real_time_deepfake_0851
python api.py &

# 2. Start the webcam pipeline with virtual camera output
python -c "
from core.webcam_pipeline import get_pipeline, PipelineConfig
from core.face_swapper import get_swapper
import cv2

config = PipelineConfig(
    camera_id=0,
    virtual_camera=True,
    virtual_camera_name='DeepFakeCam',
    frame_skip=2
)

swapper = get_swapper(det_size=(320, 320))
pipeline = get_pipeline(config)

# Load a source face
source_img = cv2.imread('path/to/your/face.jpg')
source_face = swapper.get_source_face(source_img)
pipeline.set_source_face(source_face)

pipeline.start(swapper)
print('Pipeline running - DeepFakeCam is active')
"
```

## Step 4: Configure WhatsApp Desktop

1. Open **WhatsApp Desktop**
2. Go to **Settings** (gear icon) → **Video and Voice**
3. Under **Camera**, select **DeepFakeCam**
4. Start a video call - your face will be swapped in real-time

## Step 5: Add Character AI Chat

For an interactive character experience:

1. Open the Streamlit dashboard at **http://localhost:8080**
2. Upload a character photo
3. Set a system prompt for personality
4. Toggle swap ON
5. Chat with your character - the response appears in the chat panel

## Configuration Tips

### For Better Quality
- Set `frame_skip=1` (process every frame)
- Set `enhance_quality='high'`
- Use higher resolution (1280×720)

### For Better Performance
- Set `frame_skip=3` (process every 4th frame)
- Set `enhance_quality='low'`
- Use 320×240 capture resolution
- Set `det_size=(160, 160)` for faster detection

### Smoothing
- Default: 5-frame temporal smoothing
- For less jitter: increase to 8-10 frames
- For more responsiveness: decrease to 2-3 frames

## Troubleshooting

### "No such device" when loading v4l2loopback
```bash
# Check kernel version
uname -r

# Install matching kernel headers
sudo apt install linux-headers-$(uname -r)
```

### Virtual camera not appearing in WhatsApp
```bash
# Check if the device exists
ls -la /dev/video*

# Test the virtual camera
ffplay /dev/video10
```

### Low FPS in video call
- Reduce frame processing: increase `frame_skip`
- Reduce capture resolution
- Close other CPU-intensive applications
- Try `enhance_quality='low'`

### Latency issues
- Reduce temporal smoothing window
- Decrease `frame_skip` to reduce buffering
- Use direct mode (no blending)

## Advanced: Using with Other Apps

The DeepFakeCam virtual camera works with any app that supports standard V4L2 cameras:

- **Zoom**: Settings → Video → Camera → DeepFakeCam
- **Google Meet**: Settings → Video → Camera → DeepFakeCam
- **Discord**: Voice & Video → Video Settings → Camera → DeepFakeCam
- **OBS Studio**: Add Source → Video Capture Device → DeepFakeCam
- **Telegram**: Settings → Advanced → Video calls → DeepFakeCam