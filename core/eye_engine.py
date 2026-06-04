"""
DeepFaceReal-Physics Natural Eye & Gaze Engine v2.0.0
Powered By DeathLegionTeamLK
Saccade simulation, micro-saccades, natural blink patterns,
gaze target tracking, pupil rendering, natural eye whites visible during movement.
"""
import cv2
import numpy as np
import time
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

CREDITS = "Powered By DeathLegionTeamLK"


@dataclass
class EyeEngineConfig:
    """Configuration for the Natural Eye & Gaze Engine"""
    # Blink parameters
    blink_interval_min: float = 2.0  # seconds between blinks
    blink_interval_max: float = 4.0
    blink_duration_min: float = 0.1  # seconds (100ms)
    blink_duration_max: float = 0.4  # seconds (400ms)
    
    # Saccade parameters
    saccade_interval_min: float = 0.2  # 200ms between saccades
    saccade_interval_max: float = 0.3  # 300ms
    saccade_amplitude: float = 0.15  # max eye movement amplitude (relative to eye width)
    
    # Micro-saccade parameters
    micro_saccade_interval: float = 0.05  # 50ms
    micro_saccade_amplitude: float = 0.02
    
    # Gaze parameters
    gaze_smoothing: float = 0.7
    pupil_size_min: float = 0.3
    pupil_size_max: float = 0.7
    pupil_size_rest: float = 0.5
    
    # Eye white rendering
    show_eye_whites: bool = True
    eye_white_color: Tuple[int, int, int] = (255, 255, 255)
    iris_color: Tuple[int, int, int] = (80, 60, 40)
    pupil_color: Tuple[int, int, int] = (20, 15, 10)
    
    # Performance
    fps: int = 30
    enabled: bool = True


@dataclass
class EyeState:
    """Current state of eye animation"""
    left_blink: float = 0.0  # 0=open, 1=closed
    right_blink: float = 0.0
    left_gaze_x: float = 0.0  # -1 to 1
    left_gaze_y: float = 0.0
    right_gaze_x: float = 0.0
    right_gaze_y: float = 0.0
    pupil_size: float = 0.5
    is_blinking: bool = False
    saccade_target: Tuple[float, float] = (0.0, 0.0)
    timestamp: float = 0.0


class EyeEngine:
    """
    Natural Eye & Gaze Engine simulating realistic eye movements.
    Features saccades, micro-saccades, natural blinks, and pupil rendering.
    """

    def __init__(self, config: Optional[EyeEngineConfig] = None):
        self.config = config or EyeEngineConfig()
        self._state = EyeState()
        
        # Blink state machine
        self._next_blink_time: float = time.time() + random.uniform(
            self.config.blink_interval_min, self.config.blink_interval_max
        )
        self._blink_start_time: float = 0.0
        self._blink_phase: float = 0.0
        self._blink_duration: float = 0.0
        
        # Saccade state
        self._next_saccade_time: float = time.time() + random.uniform(
            self.config.saccade_interval_min, self.config.saccade_interval_max
        )
        self._saccade_start: Tuple[float, float] = (0.0, 0.0)
        self._saccade_target: Tuple[float, float] = (0.0, 0.0)
        self._saccade_progress: float = 1.0  # 1 = done
        
        # Micro-saccade state
        self._next_micro_time: float = time.time() + self.config.micro_saccade_interval
        self._micro_offset: Tuple[float, float] = (0.0, 0.0)
        
        # Gaze
        self._gaze_target: Tuple[float, float] = (0.0, 0.0)
        self._smooth_gaze: Tuple[float, float] = (0.0, 0.0)
        
        # Timing
        self._last_time: float = time.time()
        self._frame_count: int = 0
        self._fps_timer: float = time.time()
        self._fps_count: int = 0
        self._current_fps: float = 30.0

    def update(self, force_update: bool = False) -> EyeState:
        """Update eye state with natural movements. Call every frame."""
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        self._frame_count += 1
        
        self._fps_count += 1
        if now - self._fps_timer >= 1.0:
            self._current_fps = self._fps_count / (now - self._fps_timer)
            self._fps_timer = now
            self._fps_count = 0
        
        state = EyeState(timestamp=now)
        
        # 1. Process blink
        self._process_blink(now, dt, state)
        
        # 2. Process saccades
        self._process_saccades(now, dt, state)
        
        # 3. Process micro-saccades
        self._process_micro_saccades(now, dt, state)
        
        # 4. Smooth gaze
        self._smooth_gaze = (
            self._smooth_gaze[0] * self.config.gaze_smoothing + state.left_gaze_x * (1 - self.config.gaze_smoothing),
            self._smooth_gaze[1] * self.config.gaze_smoothing + state.left_gaze_y * (1 - self.config.gaze_smoothing),
        )
        state.left_gaze_x = self._smooth_gaze[0]
        state.left_gaze_y = self._smooth_gaze[1]
        state.right_gaze_x = state.left_gaze_x
        state.right_gaze_y = state.left_gaze_y
        
        # 5. Pupil size (subtle change with gaze)
        gaze_mag = np.sqrt(state.left_gaze_x ** 2 + state.left_gaze_y ** 2)
        state.pupil_size = self.config.pupil_size_rest + gaze_mag * 0.15
        state.pupil_size = float(np.clip(state.pupil_size, self.config.pupil_size_min, self.config.pupil_size_max))
        
        # Store state
        self._state = state
        return state

    def _process_blink(self, now: float, dt: float, state: EyeState):
        """Process natural blink cycle."""
        if now >= self._next_blink_time and not state.is_blinking:
            # Start blink
            state.is_blinking = True
            self._blink_start_time = now
            self._blink_duration = random.uniform(
                self.config.blink_duration_min, self.config.blink_duration_max
            )
            self._next_blink_time = now + random.uniform(
                self.config.blink_interval_min, self.config.blink_interval_max
            ) + self._blink_duration
        
        if state.is_blinking:
            blink_elapsed = now - self._blink_start_time
            self._blink_phase = np.clip(blink_elapsed / self._blink_duration, 0.0, 1.0)
            
            # Blink curve: fast close, slightly slower open
            if self._blink_phase < 0.4:
                # Closing phase
                t = self._blink_phase / 0.4
                blink_amount = t * t  # ease-in
            elif self._blink_phase < 0.6:
                # Fully closed phase
                blink_amount = 1.0
            else:
                # Opening phase
                t = (self._blink_phase - 0.6) / 0.4
                blink_amount = 1.0 - t * (2.0 - t)  # ease-out
            
            state.left_blink = float(np.clip(blink_amount, 0.0, 1.0))
            state.right_blink = state.left_blink
            
            if self._blink_phase >= 1.0:
                state.is_blinking = False
                state.left_blink = 0.0
                state.right_blink = 0.0

    def _process_saccades(self, now: float, dt: float, state: EyeState):
        """Process rapid eye movements (saccades)."""
        if now >= self._next_saccade_time and self._saccade_progress >= 1.0:
            # Start new saccade
            self._saccade_start = (
                self._smooth_gaze[0] if hasattr(self, '_smooth_gaze') else 0.0,
                self._smooth_gaze[1] if hasattr(self, '_smooth_gaze') else 0.0,
            )
            amp = self.config.saccade_amplitude
            self._saccade_target = (
                random.uniform(-amp, amp),
                random.uniform(-amp, amp) * 0.6,  # less vertical movement
            )
            self._saccade_progress = 0.0
            self._next_saccade_time = now + random.uniform(
                self.config.saccade_interval_min, self.config.saccade_interval_max
            )
        
        if self._saccade_progress < 1.0:
            # Saccade is ballistic - very fast
            self._saccade_progress += dt * 30.0  # completes in ~33ms
            if self._saccade_progress > 1.0:
                self._saccade_progress = 1.0
            
            t = self._saccade_progress
            if t < 0.5:
                # Fast ballistic phase
                bt = t / 0.5
                fraction = bt * bt  # ease-in
            else:
                # Slow settling phase
                bt = (t - 0.5) / 0.5
                fraction = 1.0 - (1.0 - bt) * (1.0 - bt)  # ease-out
            
            state.left_gaze_x = float(self._saccade_start[0] + (self._saccade_target[0] - self._saccade_start[0]) * fraction)
            state.left_gaze_y = float(self._saccade_start[1] + (self._saccade_target[1] - self._saccade_start[1]) * fraction)
        else:
            # Hold at target with drift
            drift = np.sin(now * 0.5) * 0.005
            state.left_gaze_x = float(self._saccade_target[0] + drift)
            state.left_gaze_y = float(self._saccade_target[1])

    def _process_micro_saccades(self, now: float, dt: float, state: EyeState):
        """Process tiny involuntary eye movements during fixation."""
        if now >= self._next_micro_time:
            amp = self.config.micro_saccade_amplitude
            self._micro_offset = (
                random.uniform(-amp, amp),
                random.uniform(-amp, amp) * 0.5,
            )
            self._next_micro_time = now + self.config.micro_saccade_interval
        
        state.left_gaze_x += self._micro_offset[0]
        state.left_gaze_y += self._micro_offset[1]

    def render_eyes(self, frame: np.ndarray, face_landmarks: Optional[np.ndarray] = None,
                    eye_state: Optional[EyeState] = None) -> np.ndarray:
        """
        Render natural eyes onto face region with gaze, blink, and pupil effects.
        
        Args:
            frame: Input BGR frame
            face_landmarks: MediaPipe face mesh landmarks (468 points)
            eye_state: Current eye state (if None, uses internal state)
            
        Returns:
            Frame with rendered eye effects
        """
        if not self.config.enabled:
            return frame
        
        state = eye_state or self._state
        result = frame.copy()
        h, w = frame.shape[:2]
        
        if face_landmarks is not None and len(face_landmarks) >= 468:
            # Extract eye regions from landmarks
            # Left eye landmarks: 33-133 (inner to outer corner)
            # Right eye landmarks: 362-263 (inner to outer corner)
            
            left_eye_pts = face_landmarks[[33, 133, 157, 158, 159, 160, 161, 173, 246]].astype(np.int32)
            right_eye_pts = face_landmarks[[362, 263, 387, 386, 385, 384, 398, 466, 380]].astype(np.int32)
            
            left_eye_center = left_eye_pts.mean(axis=0).astype(np.int32)
            right_eye_center = right_eye_pts.mean(axis=0).astype(np.int32)
            
            left_eye_width = float(np.linalg.norm(face_landmarks[33] - face_landmarks[133]))
            right_eye_width = float(np.linalg.norm(face_landmarks[362] - face_landmarks[263]))
            
            self._render_single_eye(result, left_eye_center, left_eye_width, state.left_blink,
                                   state.left_gaze_x, state.left_gaze_y, state.pupil_size, left_eye_pts)
            self._render_single_eye(result, right_eye_center, right_eye_width, state.right_blink,
                                   state.right_gaze_x, state.right_gaze_y, state.pupil_size, right_eye_pts)
        
        return result

    def _render_single_eye(self, frame: np.ndarray, center: np.ndarray, eye_width: float,
                           blink: float, gaze_x: float, gaze_y: float, pupil_size: float,
                           eye_contour: np.ndarray):
        """Render a single eye with pupil, iris, and natural look."""
        if eye_width < 5:
            return
        
        # Eye dimensions
        eye_radius = int(eye_width * 0.28)
        iris_radius = int(eye_radius * 0.85)
        pupil_radius = int(iris_radius * pupil_size)
        
        # Blink effect: scale vertical dimension
        blink_factor = 1.0 - blink * 0.9  # 1.0 = open, 0.1 = closed
        
        # Eye white (sclera) - visible during movement
        if self.config.show_eye_whites:
            cv2.ellipse(
                frame,
                (int(center[0]), int(center[1])),
                (int(eye_radius), max(1, int(eye_radius * blink_factor * 0.7))),
                0, 0, 360,
                self.config.eye_white_color,
                -1
            )
        
        if blink < 0.8:  # Only render iris/pupil when eye is somewhat open
            # Gaze offset for iris/pupil
            max_offset = eye_radius * 0.3
            gaze_offset_x = int(gaze_x * max_offset)
            gaze_offset_y = int(gaze_y * max_offset)
            
            iris_center = (int(center[0]) + gaze_offset_x, int(center[1]) + gaze_offset_y)
            
            # Iris
            cv2.circle(
                frame,
                iris_center,
                max(1, int(iris_radius * blink_factor)),
                self.config.iris_color,
                -1
            )
            
            # Pupil
            cv2.circle(
                frame,
                iris_center,
                max(1, int(pupil_radius * blink_factor)),
                self.config.pupil_color,
                -1
            )
            
            # Iris detail (radial lines for realism)
            iris_detail_radius = int(iris_radius * 0.6 * blink_factor)
            for angle in range(0, 360, 30):
                rad = np.radians(angle)
                x = int(iris_center[0] + np.cos(rad) * iris_detail_radius)
                y = int(iris_center[1] + np.sin(rad) * iris_detail_radius)
                cv2.circle(frame, (x, y), 1, (60, 40, 30), -1)
            
            # Highlight (corneal reflection)
            highlight_offset = int(iris_radius * 0.25)
            highlight_x = iris_center[0] - highlight_offset
            highlight_y = iris_center[1] - highlight_offset
            cv2.circle(
                frame,
                (highlight_x, highlight_y),
                max(1, int(iris_radius * 0.2 * blink_factor)),
                (220, 220, 255),
                -1
            )
            cv2.circle(
                frame,
                (highlight_x, highlight_y),
                max(1, int(iris_radius * 0.1 * blink_factor)),
                (255, 255, 255),
                -1
            )

    def get_gaze_target(self) -> Tuple[float, float]:
        """Get current gaze target (x, y) in normalized space."""
        return self._saccade_target

    def set_gaze_target(self, target_x: float, target_y: float):
        """Set a gaze target for the eyes to look at."""
        self._saccade_target = (float(np.clip(target_x, -1.0, 1.0)),
                                float(np.clip(target_y, -1.0, 1.0)))
        self._saccade_progress = 0.0
        self._saccade_start = self._smooth_gaze

    def get_state(self) -> EyeState:
        return self._state

    def get_status(self) -> Dict[str, Any]:
        return {
            'config': {
                'blink_interval': f"{self.config.blink_interval_min}-{self.config.blink_interval_max}s",
                'blink_duration': f"{self.config.blink_duration_min}-{self.config.blink_duration_max}s",
                'saccade_amplitude': self.config.saccade_amplitude,
                'enabled': self.config.enabled,
                'show_eye_whites': self.config.show_eye_whites,
            },
            'state': {
                'is_blinking': self._state.is_blinking,
                'blink': float(f"{self._state.left_blink:.2f}"),
                'gaze': f"({self._state.left_gaze_x:.2f}, {self._state.left_gaze_y:.2f})",
                'pupil_size': float(f"{self._state.pupil_size:.2f}"),
            },
            'fps': round(self._current_fps, 1),
            'frames': self._frame_count,
            'credits': CREDITS,
        }


_eye_engine_instance: Optional[EyeEngine] = None


def get_eye_engine() -> EyeEngine:
    global _eye_engine_instance
    if _eye_engine_instance is None:
        _eye_engine_instance = EyeEngine()
    return _eye_engine_instance