import os
import cv2
import numpy as np
import time
import threading
import requests
import socket
from collections import deque
from typing import Optional, Callable, Tuple, Any
from dataclasses import dataclass, field

CREDITS = 'Powered By DeathLegionTeamLK'

try:
    import pyvirtualcam
    HAS_VIRTUAL_CAM = True
except ImportError:
    HAS_VIRTUAL_CAM = False


@dataclass
class PipelineConfig:
    camera_id: int = 0
    camera_source: str = 'usb'
    camera_url: str = ''
    capture_width: int = 640
    capture_height: int = 480
    output_width: int = 640
    output_height: int = 480
    target_fps: int = 15
    frame_skip: int = 2
    det_size: Tuple[int, int] = (320, 320)
    blend_enabled: bool = True
    enhance_enabled: bool = True
    enhance_quality: str = "medium"
    smoothing_frames: int = 5
    min_face_size: int = 50
    virtual_camera: bool = True
    virtual_camera_name: str = "DeepFakeCam"


class LandmarkSmoother:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.landmark_history: deque = deque(maxlen=window_size)
        self.bbox_history: deque = deque(maxlen=window_size)

    def smooth(self, landmarks: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if landmarks is not None:
            self.landmark_history.append(landmarks)
        if bbox is not None:
            self.bbox_history.append(bbox)
        smoothed_landmarks = landmarks
        smoothed_bbox = bbox
        if len(self.landmark_history) >= 2:
            weights = np.linspace(0.5, 1.0, len(self.landmark_history))
            weights = weights / weights.sum()
            smoothed_landmarks = np.zeros_like(self.landmark_history[0])
            for i, lm in enumerate(self.landmark_history):
                smoothed_landmarks += lm * weights[i]
        if len(self.bbox_history) >= 2:
            weights = np.linspace(0.5, 1.0, len(self.bbox_history))
            weights = weights / weights.sum()
            smoothed_bbox = np.zeros_like(self.bbox_history[0])
            for i, b in enumerate(self.bbox_history):
                smoothed_bbox += b * weights[i]
        return smoothed_landmarks, smoothed_bbox

    def reset(self):
        self.landmark_history.clear()
        self.bbox_history.clear()


def scan_ip_webcam_urls(timeout: float = 1.0) -> list:
    results = []
    common_urls = [
        'http://192.168.1.1:8080/video',
        'http://192.168.1.2:8080/video',
        'http://192.168.1.3:8080/video',
        'http://192.168.1.4:8080/video',
        'http://192.168.1.5:8080/video',
        'http://192.168.1.100:8080/video',
        'http://192.168.1.101:8080/video',
        'http://192.168.1.102:8080/video',
        'http://192.168.0.100:8080/video',
        'http://192.168.0.101:8080/video',
        'http://192.168.43.1:8080/video',
        'http://192.168.43.1:8080/video',
        'http://10.0.0.1:8080/video',
        'http://10.0.0.2:8080/video',
    ]
    for url in common_urls:
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            if r.status_code == 200:
                content_type = r.headers.get('content-type', '')
                if 'multipart/x-mixed-replace' in content_type or 'image' in content_type:
                    results.append(url)
        except Exception:
            pass
    return results


class MJPEGStreamReader:
    def __init__(self, url: str, max_buffer: int = 5):
        self.url = url
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_buffer: deque = deque(maxlen=max_buffer)
        self.lock = threading.Lock()
        self.last_frame: Optional[np.ndarray] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        stream = None
        try:
            stream = requests.get(self.url, stream=True, timeout=30)
            stream.raise_for_status()
            bytes_data = b''
            boundary = None
            content_type = stream.headers.get('content-type', '')
            if 'boundary=' in content_type:
                boundary = content_type.split('boundary=')[1].strip()
            for chunk in stream.iter_content(chunk_size=1024):
                if not self.running:
                    break
                if chunk:
                    bytes_data += chunk
                    while True:
                        start_idx = bytes_data.find(b'\xff\xd8')
                        end_idx = bytes_data.find(b'\xff\xd9')
                        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                            jpeg_data = bytes_data[start_idx:end_idx + 2]
                            bytes_data = bytes_data[end_idx + 2:]
                            nparr = np.frombuffer(jpeg_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                with self.lock:
                                    self.frame_buffer.append(frame)
                                    self.last_frame = frame
                        else:
                            break
                        if len(bytes_data) > 1000000:
                            bytes_data = b''
        except Exception as e:
            print(f"[MJPEGReader] Stream error: {e}")
        finally:
            if stream:
                stream.close()
            self.running = False

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            if len(self.frame_buffer) > 0:
                return self.frame_buffer.popleft()
            return self.last_frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def is_opened(self) -> bool:
        return self.running or len(self.frame_buffer) > 0


class WebcamPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.smoother = LandmarkSmoother(window_size=self.config.smoothing_frames)
        self.running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.frame_count = 0
        self.processed_count = 0
        self.fps_values: deque = deque(maxlen=30)
        self.current_frame: Optional[np.ndarray] = None
        self.processed_frame: Optional[np.ndarray] = None
        self.current_fps: float = 0.0
        self.source_face: Any = None
        self.swap_enabled: bool = True
        self.frame_callback: Optional[Callable] = None
        self.virtual_cam = None
        self._lock = threading.Lock()
        self._mjpeg_reader: Optional[MJPEGStreamReader] = None
        self._cap: Optional[cv2.VideoCapture] = None

    def set_source_face(self, source_face: Any):
        self.source_face = source_face
        self.smoother.reset()

    def set_swap_enabled(self, enabled: bool):
        self.swap_enabled = enabled

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def switch_camera_source(self, source_type: str, url: str = ''):
        self.config.camera_source = source_type
        self.config.camera_url = url
        was_running = self.running
        if was_running:
            self.stop()
        if was_running:
            self.start()

    def _init_camera(self) -> bool:
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._mjpeg_reader:
            self._mjpeg_reader.stop()
            self._mjpeg_reader = None

        source = self.config.camera_source

        if source == 'ip_webcam' and self.config.camera_url:
            self._mjpeg_reader = MJPEGStreamReader(self.config.camera_url)
            self._mjpeg_reader.start()
            time.sleep(0.5)
            if self._mjpeg_reader.is_opened():
                print(f"[WebcamPipeline] IP Webcam connected: {self.config.camera_url}")
                return True
            print(f"[WebcamPipeline] Failed to connect IP Webcam: {self.config.camera_url}")
            return False

        if source == 'ip_webcam':
            urls = scan_ip_webcam_urls(timeout=0.5)
            if urls:
                self.config.camera_url = urls[0]
                self._mjpeg_reader = MJPEGStreamReader(urls[0])
                self._mjpeg_reader.start()
                time.sleep(0.5)
                if self._mjpeg_reader.is_opened():
                    print(f"[WebcamPipeline] Auto-detected IP Webcam: {urls[0]}")
                    return True
            print("[WebcamPipeline] No IP Webcam found on network")
            return False

        if source == 'custom_url' and self.config.camera_url:
            self._mjpeg_reader = MJPEGStreamReader(self.config.camera_url)
            self._mjpeg_reader.start()
            time.sleep(0.5)
            if self._mjpeg_reader.is_opened():
                print(f"[WebcamPipeline] Custom URL connected: {self.config.camera_url}")
                return True
            print(f"[WebcamPipeline] Failed to connect custom URL: {self.config.camera_url}")
            return False

        cap = cv2.VideoCapture(self.config.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.capture_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        if not cap.isOpened():
            print(f"[WebcamPipeline] Failed to open camera {self.config.camera_id}")
            return False
        self._cap = cap
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[WebcamPipeline] Camera opened: {actual_width}x{actual_height}")
        return True

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                return frame
            return None
        if self._mjpeg_reader:
            return self._mjpeg_reader.read()
        return None

    def _init_virtual_camera(self):
        if not HAS_VIRTUAL_CAM or not self.config.virtual_camera:
            return
        try:
            self.virtual_cam = pyvirtualcam.Camera(
                width=self.config.output_width,
                height=self.config.output_height,
                fps=self.config.target_fps,
                backend='v4l2loopback',
                device=self.config.virtual_camera_name
            )
            print(f"[WebcamPipeline] Virtual camera started: {self.config.output_width}x{self.config.output_height}")
        except Exception as e:
            print(f"[WebcamPipeline] Virtual camera init failed: {e}")
            self.virtual_cam = None

    def _update_fps(self):
        self.fps_values.append(time.time())
        while len(self.fps_values) > 1 and (time.time() - self.fps_values[0]) > 2.0:
            self.fps_values.popleft()
        if len(self.fps_values) > 1:
            elapsed = self.fps_values[-1] - self.fps_values[0]
            self.current_fps = (len(self.fps_values) - 1) / elapsed if elapsed > 0 else 0

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.source_face is None or not self.swap_enabled:
            return frame
        if self.frame_count % (self.config.frame_skip + 1) != 0:
            return self.processed_frame if self.processed_frame is not None else frame
        if self.frame_callback:
            try:
                result = self.frame_callback(frame, self.source_face)
                return result
            except Exception as e:
                print(f"[WebcamPipeline] Frame callback error: {e}")
                return frame
        return frame

    def _pipeline_loop(self, swapper):
        if not self._init_camera():
            self.running = False
            return
        self._init_virtual_camera()
        print("[WebcamPipeline] Pipeline started")
        while self.running:
            frame = self._read_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            self.frame_count += 1
            if frame.shape[1] != self.config.capture_width or frame.shape[0] != self.config.capture_height:
                frame = cv2.resize(frame, (self.config.capture_width, self.config.capture_height))
            with self._lock:
                self.current_frame = frame.copy()
            result = self._process_frame(frame)
            with self._lock:
                self.processed_frame = result
                self._update_fps()
            if self.virtual_cam:
                try:
                    if result.shape[2] == 3:
                        rgb_frame = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_frame = result
                    if rgb_frame.shape[1] != self.config.output_width or rgb_frame.shape[0] != self.config.output_height:
                        rgb_frame = cv2.resize(rgb_frame, (self.config.output_width, self.config.output_height))
                    self.virtual_cam.send(rgb_frame)
                except Exception as e:
                    if "closed" in str(e).lower():
                        self.virtual_cam = None
            time.sleep(0.001)
        if self._cap:
            self._cap.release()
        if self._mjpeg_reader:
            self._mjpeg_reader.stop()
        if self.virtual_cam:
            self.virtual_cam.close()
        print("[WebcamPipeline] Pipeline stopped")

    def start(self, swapper=None):
        if self.running:
            return
        self.running = True
        self.processing_thread = threading.Thread(
            target=self._pipeline_loop, args=(swapper,), daemon=True
        )
        self.processing_thread.start()

    def stop(self):
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=3.0)

    def get_status(self) -> dict:
        with self._lock:
            return {
                'running': self.running,
                'fps': round(self.current_fps, 1),
                'frames_captured': self.frame_count,
                'swap_enabled': self.swap_enabled,
                'source_set': self.source_face is not None,
                'camera_source': self.config.camera_source,
                'camera_url': self.config.camera_url,
                'config': {
                    'camera': self.config.camera_id,
                    'source': self.config.camera_source,
                    'resolution': f"{self.config.capture_width}x{self.config.capture_height}",
                    'frame_skip': self.config.frame_skip,
                    'enhance_quality': self.config.enhance_quality,
                    'blend_enabled': self.config.blend_enabled,
                }
            }

    def get_current_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            return self.current_frame, self.processed_frame


_pipeline_instance: Optional[WebcamPipeline] = None


def get_pipeline() -> WebcamPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = WebcamPipeline()
    return _pipeline_instance