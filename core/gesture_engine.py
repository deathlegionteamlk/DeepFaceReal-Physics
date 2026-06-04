"""
DeepFaceReal-Physics Conversational Gesture Engine v2.0.0
Powered By DeathLegionTeamLK
Hand gesture patterns matching speech rhythm, shoulder/head micro-movements,
posture variations, speech-driven gesture timing, configurable intensity.
"""
import cv2
import numpy as np
import time
import random
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from collections import deque

CREDITS = "Powered By DeathLegionTeamLK"


@dataclass
class GestureConfig:
    """Configuration for the Gesture Engine"""
    # Gesture parameters
    gesture_intensity: float = 0.7
    gesture_frequency: float = 1.0  # gestures per second during speech
    
    # Hand gesture parameters
    hand_gesture_amplitude: float = 0.15  # relative to frame width
    hand_gesture_speed: float = 0.8
    
    # Shoulder/head micro-movements
    shoulder_movement_amplitude: float = 0.03
    shoulder_movement_frequency: float = 0.5
    
    # Posture parameters
    posture_variation: float = 0.05
    posture_change_interval: float = 3.0  # seconds
    
    # Speech-driven parameters
    emphasis_gesture_threshold: float = 0.6
    gesture_sustain_time: float = 0.3  # seconds
    
    # Performance
    fps: int = 30
    enabled: bool = True


@dataclass
class GestureFrame:
    """Single frame of gesture output"""
    hand_left_pos: Tuple[float, float] = (0.0, 0.0)  # -1 to 1 normalized
    hand_right_pos: Tuple[float, float] = (0.0, 0.0)
    hand_left_openness: float = 0.5  # 0=closed fist, 1=open hand
    hand_right_openness: float = 0.5
    shoulder_tilt: float = 0.0  # -1 to 1
    shoulder_raise: float = 0.0
    head_tilt: float = 0.0
    posture_lean: float = 0.0
    gesture_active: bool = False
    gesture_type: str = "idle"  # idle, emphatic, conversational, thinking
    emphasis_gesture: bool = False
    timestamp: float = 0.0


class GesturePattern:
    """Predefined gesture patterns that can be triggered."""

    @staticmethod
    def emphatic_point(intensity: float = 1.0) -> Dict[str, float]:
        return {
            'hand_right_x': 0.3 * intensity, 'hand_right_y': -0.2 * intensity,
            'hand_left_x': -0.1 * intensity, 'hand_left_y': 0.0,
            'hand_right_openness': 0.9, 'hand_left_openness': 0.4,
            'shoulder_tilt': 0.1 * intensity, 'head_tilt': 0.05 * intensity,
        }

    @staticmethod
    def conversational_open(intensity: float = 1.0) -> Dict[str, float]:
        return {
            'hand_right_x': 0.2 * intensity, 'hand_right_y': 0.1 * intensity,
            'hand_left_x': -0.2 * intensity, 'hand_left_y': 0.1 * intensity,
            'hand_right_openness': 0.7, 'hand_left_openness': 0.7,
            'shoulder_tilt': 0.0, 'head_tilt': 0.02 * intensity,
        }

    @staticmethod
    def thinking(intensity: float = 1.0) -> Dict[str, float]:
        return {
            'hand_right_x': 0.15 * intensity, 'hand_right_y': -0.3 * intensity,
            'hand_left_x': -0.1 * intensity, 'hand_left_y': -0.1 * intensity,
            'hand_right_openness': 0.3, 'hand_left_openness': 0.5,
            'shoulder_tilt': 0.05 * intensity, 'head_tilt': 0.1 * intensity,
        }

    @staticmethod
    def resting() -> Dict[str, float]:
        return {
            'hand_right_x': 0.1, 'hand_right_y': 0.0,
            'hand_left_x': -0.1, 'hand_left_y': 0.0,
            'hand_right_openness': 0.4, 'hand_left_openness': 0.4,
            'shoulder_tilt': 0.0, 'head_tilt': 0.0,
        }


class GestureEngine:
    """
    Conversational Gesture Engine producing natural body movements
    synchronized with speech rhythm and emphasis.
    """

    def __init__(self, config: Optional[GestureConfig] = None):
        self.config = config or GestureConfig()
        self._patterns = GesturePattern()
        
        # State
        self._frame_count: int = 0
        self._last_time: float = time.time()
        self._current_fps: float = 30.0
        
        # Smoothing buffers
        self._smooth_hand_left: Tuple[float, float] = (0.0, 0.0)
        self._smooth_hand_right: Tuple[float, float] = (0.0, 0.0)
        self._smooth_hand_left_open: float = 0.4
        self._smooth_hand_right_open: float = 0.4
        self._smooth_shoulder_tilt: float = 0.0
        self._smooth_head_tilt: float = 0.0
        self._smooth_posture: float = 0.0
        
        # Gesture timing
        self._gesture_queue: List[Dict] = []
        self._current_gesture: Optional[Dict] = None
        self._gesture_start_time: float = 0.0
        self._gesture_duration: float = 0.0
        self._gesture_phase: float = 0.0
        
        # Natural oscillation phases
        self._idle_phase: float = random.uniform(0, 2 * math.pi)
        self._shoulder_phase: float = random.uniform(0, 2 * math.pi)
        self._posture_phase: float = random.uniform(0, 2 * math.pi)
        self._last_posture_change: float = time.time()
        self._posture_target: float = 0.0
        
        # Speech emphasis tracking
        self._emphasis_energy: float = 0.0
        self._emphasis_sustain: float = 0.0
        self._last_emphasis_time: float = 0.0
        
        # FPS tracking
        self._fps_count: int = 0
        self._fps_timer: float = time.time()

    def update(self, emphasis: float = 0.0, is_speaking: bool = False,
               audio_energy: float = 0.0, audio_pitch: float = 0.0) -> GestureFrame:
        """Update gesture state. Call every frame."""
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        self._frame_count += 1
        
        # FPS tracking
        self._fps_count += 1
        if now - self._fps_timer >= 1.0:
            self._current_fps = self._fps_count / (now - self._fps_timer)
            self._fps_timer = now
            self._fps_count = 0
        
        frame = GestureFrame(timestamp=now)
        
        # Track emphasis from audio
        self._emphasis_energy = audio_energy
        if emphasis > self.config.emphasis_gesture_threshold:
            self._emphasis_sustain = self.config.gesture_sustain_time
            if now - self._last_emphasis_time > 0.5:  # min gap between emphatic gestures
                self._queue_emphatic_gesture(emphasis)
                self._last_emphasis_time = now
        
        if self._emphasis_sustain > 0:
            self._emphasis_sustain -= dt
        
        # Process gesture queue
        self._process_gesture_queue(now, dt)
        
        # Natural body motion
        self._update_idle_motion(now, dt, frame)
        
        # Blend current gesture with idle motion
        self._blend_gesture_with_idle(frame, is_speaking)
        
        # Update posture
        self._update_posture(now, dt, frame)
        
        # Smooth all outputs
        self._smooth_outputs(frame)
        
        return frame

    def _queue_emphatic_gesture(self, emphasis: float):
        """Queue an emphatic gesture based on speech emphasis."""
        gesture_type = random.choice(['point', 'conversational'])
        intensity = np.clip(emphasis * 1.5, 0.5, 1.0)
        
        if gesture_type == 'point':
            params = self._patterns.emphatic_point(intensity)
            params['type'] = 'emphatic'
            params['duration'] = 0.4 + random.uniform(0, 0.3)
        else:
            params = self._patterns.conversational_open(intensity)
            params['type'] = 'conversational'
            params['duration'] = 0.5 + random.uniform(0, 0.4)
        
        self._gesture_queue.append(params)

    def _process_gesture_queue(self, now: float, dt: float):
        """Process the current gesture animation."""
        if self._current_gesture is None and self._gesture_queue:
            self._current_gesture = self._gesture_queue.pop(0)
            self._gesture_start_time = now
            self._gesture_duration = self._current_gesture['duration']
            self._gesture_phase = 0.0
        
        if self._current_gesture is not None:
            dt_local = now - self._gesture_start_time
            self._gesture_phase = np.clip(dt_local / self._gesture_duration, 0.0, 1.0)
            
            if self._gesture_phase >= 1.0:
                self._current_gesture = None

    def _update_idle_motion(self, now: float, dt: float, frame: GestureFrame):
        """Update natural idle body motion."""
        self._idle_phase += dt * self.config.gesture_frequency * 2 * math.pi
        
        # Subtle hand drift during idle
        idle_amp = 0.3 * self.config.gesture_intensity * self.config.hand_gesture_amplitude
        frame.hand_left_pos = (
            math.sin(self._idle_phase * 0.7) * idle_amp,
            math.cos(self._idle_phase * 0.5) * idle_amp * 0.5,
        )
        frame.hand_right_pos = (
            math.sin(self._idle_phase * 0.7 + 2.0) * idle_amp,
            math.cos(self._idle_phase * 0.5 + 2.0) * idle_amp * 0.5,
        )
        
        # Shoulder micro-movements
        self._shoulder_phase += dt * self.config.shoulder_movement_frequency * 2 * math.pi
        shoulder_amp = self.config.shoulder_movement_amplitude * self.config.gesture_intensity
        frame.shoulder_tilt = math.sin(self._shoulder_phase) * shoulder_amp
        frame.shoulder_raise = math.cos(self._shoulder_phase * 0.6) * shoulder_amp * 0.5
        
        # Head micro-tilt
        frame.head_tilt = math.sin(self._idle_phase * 0.3) * 0.02 * self.config.gesture_intensity
        
        # Hand openness during idle
        frame.hand_left_openness = 0.4 + 0.1 * math.sin(self._idle_phase * 0.4)
        frame.hand_right_openness = 0.4 + 0.1 * math.sin(self._idle_phase * 0.4 + 1.0)

    def _blend_gesture_with_idle(self, frame: GestureFrame, is_speaking: bool):
        """Blend active gesture with idle motion."""
        if self._current_gesture is None:
            frame.gesture_active = False
            frame.gesture_type = "idle"
            return
        
        gest = self._current_gesture
        phase = self._gesture_phase
        
        # Ease-in / ease-out curve
        if phase < 0.2:
            blend = phase / 0.2
            blend = blend * blend  # ease-in
        elif phase > 0.8:
            blend = (1.0 - phase) / 0.2
            blend = 1.0 - (1.0 - blend) * (1.0 - blend)  # ease-out
        else:
            blend = 1.0
        
        intensity = self.config.gesture_intensity * blend
        
        # Blend hand positions
        frame.hand_right_pos = (
            frame.hand_right_pos[0] * (1 - blend) + gest.get('hand_right_x', 0) * intensity,
            frame.hand_right_pos[1] * (1 - blend) + gest.get('hand_right_y', 0) * intensity,
        )
        frame.hand_left_pos = (
            frame.hand_left_pos[0] * (1 - blend) + gest.get('hand_left_x', 0) * intensity,
            frame.hand_left_pos[1] * (1 - blend) + gest.get('hand_left_y', 0) * intensity,
        )
        
        frame.hand_right_openness = frame.hand_right_openness * (1 - blend) + gest.get('hand_right_openness', 0.5) * intensity
        frame.hand_left_openness = frame.hand_left_openness * (1 - blend) + gest.get('hand_left_openness', 0.5) * intensity
        
        frame.shoulder_tilt = frame.shoulder_tilt * (1 - blend) + gest.get('shoulder_tilt', 0) * intensity
        frame.head_tilt = frame.head_tilt * (1 - blend) + gest.get('head_tilt', 0) * intensity
        
        frame.gesture_active = blend > 0.3
        frame.gesture_type = gest.get('type', 'conversational')
        frame.emphasis_gesture = gest.get('type') == 'emphatic'

    def _update_posture(self, now: float, dt: float, frame: GestureFrame):
        """Update posture with slow variations."""
        if now - self._last_posture_change > self.config.posture_change_interval:
            self._posture_target = random.uniform(-1, 1) * self.config.posture_variation
            self._last_posture_change = now
        
        # Smooth posture transition
        self._posture_phase += dt * 0.3
        posture_drift = math.sin(self._posture_phase * 0.5) * self.config.posture_variation * 0.3
        frame.posture_lean = self._posture_target + posture_drift

    def _smooth_outputs(self, frame: GestureFrame):
        """Apply temporal smoothing to all outputs."""
        smoothing = 0.85
        
        self._smooth_hand_left = (
            self._smooth_hand_left[0] * smoothing + frame.hand_left_pos[0] * (1 - smoothing),
            self._smooth_hand_left[1] * smoothing + frame.hand_left_pos[1] * (1 - smoothing),
        )
        self._smooth_hand_right = (
            self._smooth_hand_right[0] * smoothing + frame.hand_right_pos[0] * (1 - smoothing),
            self._smooth_hand_right[1] * smoothing + frame.hand_right_pos[1] * (1 - smoothing),
        )
        self._smooth_hand_left_open = self._smooth_hand_left_open * smoothing + frame.hand_left_openness * (1 - smoothing)
        self._smooth_hand_right_open = self._smooth_hand_right_open * smoothing + frame.hand_right_openness * (1 - smoothing)
        self._smooth_shoulder_tilt = self._smooth_shoulder_tilt * smoothing + frame.shoulder_tilt * (1 - smoothing)
        self._smooth_head_tilt = self._smooth_head_tilt * smoothing + frame.head_tilt * (1 - smoothing)
        
        frame.hand_left_pos = self._smooth_hand_left
        frame.hand_right_pos = self._smooth_hand_right
        frame.hand_left_openness = self._smooth_hand_left_open
        frame.hand_right_openness = self._smooth_hand_right_open
        frame.shoulder_tilt = self._smooth_shoulder_tilt
        frame.head_tilt = self._smooth_head_tilt

    def render_gesture_overlay(self, frame: np.ndarray, gesture: Optional[GestureFrame] = None,
                               hand_landmarks: Optional[Dict] = None) -> np.ndarray:
        """Render gesture visualization overlay."""
        if not self.config.enabled:
            return frame
        
        result = frame.copy()
        h, w = frame.shape[:2]
        g = gesture or self.update()
        
        if hand_landmarks and 'left' in hand_landmarks:
            # Draw detected hand landmarks if available
            for pt in hand_landmarks['left']:
                cv2.circle(result, (int(pt[0] * w), int(pt[1] * h)), 3, (0, 255, 0), -1)
        if hand_landmarks and 'right' in hand_landmarks:
            for pt in hand_landmarks['right']:
                cv2.circle(result, (int(pt[0] * w), int(pt[1] * h)), 3, (0, 200, 255), -1)
        
        # Draw gesture state info
        cx, cy = w // 2, h - 60
        cv2.putText(result, f"Gesture: {g.gesture_type}", (10, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)
        cv2.putText(result, f"Active: {'Yes' if g.gesture_active else 'No'} | "
                    f"Intensity: {self.config.gesture_intensity:.1f}",
                    (10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return result

    def get_status(self) -> Dict[str, Any]:
        return {
            'config': {
                'gesture_intensity': self.config.gesture_intensity,
                'gesture_frequency': self.config.gesture_frequency,
                'emphasis_threshold': self.config.emphasis_gesture_threshold,
                'enabled': self.config.enabled,
            },
            'state': {
                'gesture_type': self._current_gesture.get('type', 'idle') if self._current_gesture else 'idle',
                'queue_size': len(self._gesture_queue),
                'emphasis_sustain': round(self._emphasis_sustain, 2),
            },
            'fps': round(self._current_fps, 1),
            'frames': self._frame_count,
            'credits': CREDITS,
        }


_gesture_engine_instance: Optional[GestureEngine] = None


def get_gesture_engine() -> GestureEngine:
    global _gesture_engine_instance
    if _gesture_engine_instance is None:
        _gesture_engine_instance = GestureEngine()
    return _gesture_engine_instance