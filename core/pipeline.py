"""
DeepFaceReal-Physics Real-Time Pipeline Integration v2.0.0
Powered By DeathLegionTeamLK
Wires all engines into single async pipeline:
Audio -> Features -> TalkingHead -> LipSync -> EyeEngine -> GestureEngine -> Composite
CPU-optimized: frame skipping, cached inference, async queues.
"""
import cv2
import numpy as np
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List, Tuple
from collections import deque
from enum import Enum

CREDITS = "Powered By DeathLegionTeamLK"

from core.face_3d_engine import get_face_3d_engine, Face3DEngine, Face3DData
from core.talking_head import get_talking_head, TalkingHeadEngine, MotionFrame, AudioFeatures
from core.lip_sync import create_lip_sync, EnhancedLipSync, LipSyncConfig
from core.eye_engine import get_eye_engine, EyeEngine, EyeState
from core.gesture_engine import get_gesture_engine, GestureEngine, GestureFrame


class PipelineStage(Enum):
    """Pipeline processing stages"""
    IDLE = "idle"
    AUDIO_FEATURES = "audio_features"
    TALKING_HEAD = "talking_head"
    LIP_SYNC = "lip_sync"
    EYE_GAZE = "eye_gaze"
    GESTURES = "gestures"
    COMPOSITE = "composite"
    COMPLETE = "complete"


@dataclass
class PipelineConfig:
    """Configuration for the Real-Time Pipeline"""
    # Resolution
    output_width: int = 640
    output_height: int = 480
    
    # Frame skipping for CPU optimization
    frame_skip_3d: int = 2  # Process 3D face every Nth frame
    frame_skip_lip: int = 1
    frame_skip_eye: int = 0
    frame_skip_gesture: int = 0
    
    # Cache settings
    enable_cache: bool = True
    cache_max_size: int = 30
    
    # Async pipeline
    use_async: bool = True
    max_queue_size: int = 10
    
    # Performance targets
    target_fps: float = 20.0
    min_frame_interval: float = 0.033  # ~30fps max
    
    # Debug
    show_stage_timing: bool = False
    show_pipeline_metrics: bool = True
    
    # Credits
    credits: str = CREDITS


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    fps: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    frames_processed: int = 0
    frames_skipped: int = 0
    average_latency: float = 0.0
    pipeline_stage: str = "idle"
    credits: str = CREDITS


class PipelineFrame:
    """Frame data passing through pipeline stages"""
    def __init__(self, frame: np.ndarray, timestamp: float, audio_chunk: Optional[np.ndarray] = None):
        self.original_frame = frame.copy()
        self.current_frame = frame.copy()
        self.timestamp = timestamp
        self.audio_chunk = audio_chunk
        self.frame_number: int = 0
        
        # Stage outputs
        self.face_3d_data: Optional[Face3DData] = None
        self.motion_frame: Optional[MotionFrame] = None
        self.lip_synced: bool = False
        self.eye_state: Optional[EyeState] = None
        self.gesture_frame: Optional[GestureFrame] = None
        
        # Stage tracking
        self.stages_completed: List[PipelineStage] = []
        self.stage_times: Dict[str, float] = {}
        self.current_stage: PipelineStage = PipelineStage.IDLE


class RealTimePipeline:
    """
    Real-Time Pipeline that wires all engines together.
    Audio -> 3D Face -> TalkingHead -> LipSync -> EyeEngine -> GestureEngine -> Composite
    CPU-optimized with frame skipping, caching, and async queues.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize all engines
        self.face_3d = get_face_3d_engine()
        self.talking_head = get_talking_head()
        self.lip_sync = create_lip_sync()
        self.eye_engine = get_eye_engine()
        self.gesture_engine = get_gesture_engine()
        
        # Pipeline state
        self._running: bool = False
        self._frame_count: int = 0
        self._frames_skipped: int = 0
        self._last_time: float = time.time()
        self._start_time: float = time.time()
        
        # Performance tracking
        self._fps_times: deque = deque(maxlen=60)
        self._stage_times: Dict[str, deque] = {
            'face_3d': deque(maxlen=30),
            'talking_head': deque(maxlen=30),
            'lip_sync': deque(maxlen=30),
            'eye_engine': deque(maxlen=30),
            'gesture_engine': deque(maxlen=30),
            'composite': deque(maxlen=30),
            'total': deque(maxlen=30),
        }
        
        # Frame skipping counters
        self._skip_counters = {
            'face_3d': 0,
            'lip_sync': 0,
            'eye_engine': 0,
            'gesture_engine': 0,
        }
        
        # Async queues
        self._input_queue: queue.Queue = queue.Queue(maxsize=self.config.max_queue_size)
        self._output_queue: queue.Queue = queue.Queue(maxsize=self.config.max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        
        # Cache for repeated phonemes
        self._cache: Dict[str, np.ndarray] = {}
        
        # Audio state
        self._audio_buffer: deque = deque(maxlen=16000)  # 1 second at 16kHz
        
        # Callbacks
        self._on_frame: Optional[Callable] = None

    def set_config(self, config: PipelineConfig):
        """Update pipeline configuration."""
        self.config = config
        self.face_3d.config.frame_skip = config.frame_skip_3d

    def process_frame(self, frame: np.ndarray,
                      audio_chunk: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process a single frame through the full pipeline.
        
        Pipeline: Audio -> 3D Face -> TalkingHead -> LipSync -> EyeEngine -> GestureEngine -> Composite
        
        Args:
            frame: Input BGR frame
            audio_chunk: Optional audio chunk for lip sync
            
        Returns:
            Processed output frame
        """
        start_total = time.perf_counter()
        now = time.time()
        self._frame_count += 1
        
        pf = PipelineFrame(frame, now, audio_chunk)
        h, w = frame.shape[:2]
        
        # Resize if needed
        if w != self.config.output_width or h != self.config.output_height:
            pf.current_frame = cv2.resize(pf.current_frame,
                                          (self.config.output_width, self.config.output_height))
        
        # Stage 1: 3D Face Engine
        t0 = time.perf_counter()
        self._skip_counters['face_3d'] += 1
        if self._skip_counters['face_3d'] > self.config.frame_skip_3d:
            self._skip_counters['face_3d'] = 0
            pf.face_3d_data = self.face_3d.process_frame(pf.current_frame)
        else:
            pf.face_3d_data = self.face_3d._get_cached_data(h, w, now) if self.face_3d._cached_data else None
        pf.stages_completed.append(PipelineStage.AUDIO_FEATURES)
        dt = time.perf_counter() - t0
        self._stage_times['face_3d'].append(dt)
        
        # Stage 2: Audio -> TalkingHead
        t0 = time.perf_counter()
        if audio_chunk is not None and len(audio_chunk) > 0:
            self._audio_buffer.extend(audio_chunk.tolist())
            motion = self.talking_head.process_audio(audio_chunk, 16000)
        else:
            silence = np.zeros(800, dtype=np.float32)
            motion = self.talking_head.process_audio(silence, 16000)
        pf.motion_frame = motion
        pf.stages_completed.append(PipelineStage.TALKING_HEAD)
        dt = time.perf_counter() - t0
        self._stage_times['talking_head'].append(dt)
        
        # Stage 3: Lip Sync
        t0 = time.perf_counter()
        self._skip_counters['lip_sync'] += 1
        face_lms = pf.face_3d_data.landmarks_2d if pf.face_3d_data and pf.face_3d_data.face_detected else None
        if self._skip_counters['lip_sync'] > self.config.frame_skip_lip:
            self._skip_counters['lip_sync'] = 0
            pf.current_frame = self.lip_sync.sync_frame(
                pf.current_frame, audio_chunk, face_lms
            )
        pf.lip_synced = True
        pf.stages_completed.append(PipelineStage.LIP_SYNC)
        dt = time.perf_counter() - t0
        self._stage_times['lip_sync'].append(dt)
        
        # Stage 4: Eye Engine
        t0 = time.perf_counter()
        self._skip_counters['eye_engine'] += 1
        eye_state = self.eye_engine.update()
        pf.eye_state = eye_state
        if self._skip_counters['eye_engine'] > self.config.frame_skip_eye:
            face_lms_3d = pf.face_3d_data.landmarks_3d if pf.face_3d_data and pf.face_3d_data.face_detected else None
            if face_lms_3d is not None:
                pf.current_frame = self.eye_engine.render_eyes(pf.current_frame, face_lms_3d, eye_state)
        pf.stages_completed.append(PipelineStage.EYE_GAZE)
        dt = time.perf_counter() - t0
        self._stage_times['eye_engine'].append(dt)
        
        # Stage 5: Gesture Engine
        t0 = time.perf_counter()
        self._skip_counters['gesture_engine'] += 1
        emphasis = motion.emphasis if motion else 0.0
        energy = 0.0
        if audio_chunk is not None and len(audio_chunk) > 0:
            energy = float(np.abs(audio_chunk).mean())
        gesture = self.gesture_engine.update(
            emphasis=emphasis, is_speaking=emphasis > 0.3,
            audio_energy=energy
        )
        pf.gesture_frame = gesture
        pf.stages_completed.append(PipelineStage.GESTURES)
        dt = time.perf_counter() - t0
        self._stage_times['gesture_engine'].append(dt)
        
        # Stage 6: Composite (add overlays)
        t0 = time.perf_counter()
        pf.current_frame = self._composite_frame(pf)
        pf.stages_completed.append(PipelineStage.COMPOSITE)
        dt = time.perf_counter() - t0
        self._stage_times['composite'].append(dt)
        
        # Performance tracking
        total_time = time.perf_counter() - start_total
        self._stage_times['total'].append(total_time)
        self._fps_times.append(now)
        
        # Clean old fps entries
        cutoff = now - 1.0
        while self._fps_times and self._fps_times[0] < cutoff:
            self._fps_times.popleft()
        
        return pf.current_frame

    def _composite_frame(self, pf: PipelineFrame) -> np.ndarray:
        """Composite all overlays onto the frame."""
        result = pf.current_frame.copy()
        h, w = result.shape[:2]
        
        # Add 3D face mesh overlay (subtle)
        if pf.face_3d_data and pf.face_3d_data.face_detected:
            result = self.face_3d.render_mesh_overlay(
                result, pf.face_3d_data,
                show_landmarks=False, show_triangles=False, show_pose=False
            )
        
        # Add HUD with pipeline info
        if self.config.show_pipeline_metrics:
            fps = len(self._fps_times) / max(time.time() - self._fps_times[0], 0.01) if self._fps_times else 0
            avg_latency = float(np.mean(list(self._stage_times['total']))) * 1000 if self._stage_times['total'] else 0
            
            info_lines = [
                f"FPS: {fps:.1f}",
                f"Latency: {avg_latency:.1f}ms",
                f"Frame: {self._frame_count}",
                f"Stages: {len(pf.stages_completed)}/6",
            ]
            
            if pf.motion_frame:
                m = pf.motion_frame
                if m.emphasis > 0.5:
                    info_lines.append(f"Emphasis: {'!' * int(m.emphasis * 5)}")
                if m.is_nodding:
                    info_lines.append(f"Nodding: Yes")
            
            y_offset = 10
            for line in info_lines:
                cv2.putText(result, line, (w - 200, y_offset + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)
                y_offset += 18
        
        # Credits
        cv2.putText(result, CREDITS, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        
        return result

    def start(self):
        """Start the pipeline (for use with async processing)."""
        self._running = True
        self._start_time = time.time()
        if self.config.use_async:
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
        print(f"[Pipeline] Started. Target FPS: {self.config.target_fps}")

    def stop(self):
        """Stop the pipeline."""
        self._running = False
        print(f"[Pipeline] Stopped. Processed {self._frame_count} frames.")

    def _worker_loop(self):
        """Async worker thread for pipeline processing."""
        while self._running:
            try:
                frame_data = self._input_queue.get(timeout=0.1)
                if frame_data is None:
                    break
                frame, audio = frame_data
                result = self.process_frame(frame, audio)
                self._output_queue.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Pipeline] Worker error: {e}")

    def push_frame(self, frame: np.ndarray, audio: Optional[np.ndarray] = None):
        """Push frame into async pipeline."""
        if self.config.use_async and self._running:
            try:
                self._input_queue.put_nowait((frame, audio))
            except queue.Full:
                self._frames_skipped += 1
        else:
            self.process_frame(frame, audio)

    def pop_frame(self) -> Optional[np.ndarray]:
        """Get processed frame from async pipeline output."""
        try:
            return self._output_queue.get_nowait()
        except queue.Empty:
            return None

    def get_metrics(self) -> PipelineMetrics:
        """Get current pipeline performance metrics."""
        metrics = PipelineMetrics()
        now = time.time()
        
        if self._fps_times:
            metrics.fps = len(self._fps_times) / max(now - self._fps_times[0], 0.01)
        
        for stage_name, times in self._stage_times.items():
            if times:
                metrics.stage_times[stage_name] = float(np.mean(list(times)) * 1000)
        
        metrics.frames_processed = self._frame_count
        metrics.frames_skipped = self._frames_skipped
        
        if self._stage_times['total']:
            metrics.average_latency = float(np.mean(list(self._stage_times['total'])) * 1000)
        
        metrics.pipeline_stage = "running" if self._running else "stopped"
        return metrics

    def get_status(self) -> Dict[str, Any]:
        """Get full pipeline status."""
        metrics = self.get_metrics()
        status = {
            'running': self._running,
            'uptime': time.time() - self._start_time,
            'metrics': {
                'fps': round(metrics.fps, 1),
                'frames_processed': metrics.frames_processed,
                'frames_skipped': metrics.frames_skipped,
                'average_latency_ms': round(metrics.average_latency, 2),
                'stage_times_ms': {k: round(v, 2) for k, v in metrics.stage_times.items()},
            },
            'config': {
                'output_resolution': f"{self.config.output_width}x{self.config.output_height}",
                'frame_skip_3d': self.config.frame_skip_3d,
                'frame_skip_lip': self.config.frame_skip_lip,
                'use_async': self.config.use_async,
                'target_fps': self.config.target_fps,
            },
            'engines': {
                'face_3d': self.face_3d.get_status(),
                'talking_head': self.talking_head.get_status(),
                'lip_sync': self.lip_sync.get_status(),
                'eye_engine': self.eye_engine.get_status(),
                'gesture_engine': self.gesture_engine.get_status(),
            },
            'credits': CREDITS,
        }
        return status


_pipeline_instance: Optional[RealTimePipeline] = None


def get_pipeline() -> RealTimePipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RealTimePipeline()
    return _pipeline_instance