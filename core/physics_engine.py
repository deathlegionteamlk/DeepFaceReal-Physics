import cv2
import numpy as np
import mediapipe as mp
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any
from collections import deque

CREDITS = 'Powered By DeathLegionTeamLK'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


@dataclass
class PhysicsConfig:
    mass: float = 1.0
    damping: float = 0.85
    spring_stiffness: float = 0.3
    friction: float = 0.92
    gravity: float = 0.02
    smoothing_alpha: float = 0.6
    intensity: float = 1.0
    enabled: bool = True


@dataclass
class LandmarkData:
    pose: np.ndarray = None
    left_hand: np.ndarray = None
    right_hand: np.ndarray = None
    face_mesh: np.ndarray = None
    timestamp: float = 0.0
    all_landmarks_count: int = 0

    def to_dict(self) -> Dict:
        d = {'timestamp': self.timestamp, 'all_landmarks_count': self.all_landmarks_count}
        if self.pose is not None:
            d['pose'] = self.pose.tolist()
        if self.left_hand is not None:
            d['left_hand'] = self.left_hand.tolist()
        if self.right_hand is not None:
            d['right_hand'] = self.right_hand.tolist()
        if self.face_mesh is not None:
            d['face_mesh'] = self.face_mesh.tolist()
        return d


@dataclass
class PhysicsState:
    velocities: Dict[str, np.ndarray] = field(default_factory=dict)
    positions: Dict[str, np.ndarray] = field(default_factory=dict)
    spring_positions: Dict[str, float] = field(default_factory=dict)
    spring_velocities: Dict[str, float] = field(default_factory=dict)
    smoothed_landmarks: Dict[str, np.ndarray] = field(default_factory=dict)
    raw_landmarks: Dict[str, np.ndarray] = field(default_factory=dict)
    head_pose: Dict[str, float] = field(default_factory=lambda: {'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0})
    hand_positions: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {'left': (0.0, 0.0), 'right': (0.0, 0.0)})
    blink_state: float = 0.0
    micro_expression_offset: float = 0.0
    head_bob_phase: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    ear_value: float = 0.35

    def get_landmark_count(self) -> int:
        total = 0
        for k, v in self.raw_landmarks.items():
            if v is not None:
                total += len(v)
        return total


EAR_THRESHOLD = 0.20
BLINK_COOLDOWN = 0.3


class HolisticTracker:
    def __init__(self, config: Optional[PhysicsConfig] = None):
        self.config = config or PhysicsConfig()
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.state = PhysicsState()
        self.frame_skip_counter = 0
        self.frame_skip = 2
        self.last_process_time = 0.0
        self.fps_history = deque(maxlen=30)
        self.last_blink_time = 0.0
        self.micro_expression_phase = 0.0
        self._init_springs()

    def _init_springs(self):
        spring_keys = ['head_tilt_x', 'head_tilt_y', 'head_bob', 'shoulder_sway']
        for k in spring_keys:
            self.state.spring_positions[k] = 0.0
            self.state.spring_velocities[k] = 0.0

    def process_frame(self, frame: np.ndarray) -> LandmarkData:
        self.frame_skip_counter += 1
        if self.frame_skip_counter % (self.frame_skip + 1) != 0:
            return self._get_last_landmark_data()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        h, w = frame.shape[:2]
        now = time.time()

        if self.last_process_time > 0:
            dt = now - self.last_process_time
            if dt > 0:
                fps_val = 1.0 / dt
                self.fps_history.append(fps_val)
                self.state.fps = sum(self.fps_history) / max(len(self.fps_history), 1)
        self.last_process_time = now

        data = LandmarkData(timestamp=now)
        self.state.frame_count += 1

        if results.pose_landmarks:
            pts = np.array([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark])
            self.state.raw_landmarks['pose'] = pts
            data.pose = self._normalize_landmarks(pts, w, h)
            self._update_head_pose(pts)
        else:
            data.pose = None

        if results.left_hand_landmarks:
            pts = np.array([(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark])
            self.state.raw_landmarks['left_hand'] = pts
            data.left_hand = self._normalize_landmarks(pts, w, h)
            self._update_hand_position('left', pts)
        else:
            data.left_hand = None

        if results.right_hand_landmarks:
            pts = np.array([(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark])
            self.state.raw_landmarks['right_hand'] = pts
            data.right_hand = self._normalize_landmarks(pts, w, h)
            self._update_hand_position('right', pts)
        else:
            data.right_hand = None

        if results.face_landmarks:
            pts = np.array([(lm.x, lm.y, lm.z) for lm in results.face_landmarks.landmark])
            self.state.raw_landmarks['face_mesh'] = pts
            data.face_mesh = self._normalize_landmarks(pts, w, h)
            self._detect_blink(pts)
        else:
            data.face_mesh = None

        data.all_landmarks_count = self.state.get_landmark_count()

        if self.config.enabled:
            self._apply_physics()

        self._update_micro_expressions()
        self._temporal_smoothing(data)

        return data

    def _get_last_landmark_data(self) -> LandmarkData:
        data = LandmarkData(timestamp=time.time())
        if 'pose' in self.state.raw_landmarks and self.state.raw_landmarks['pose'] is not None:
            data.pose = self.state.raw_landmarks['pose']
        if 'left_hand' in self.state.raw_landmarks:
            data.left_hand = self.state.raw_landmarks.get('left_hand')
        if 'right_hand' in self.state.raw_landmarks:
            data.right_hand = self.state.raw_landmarks.get('right_hand')
        if 'face_mesh' in self.state.raw_landmarks:
            data.face_mesh = self.state.raw_landmarks.get('face_mesh')
        data.all_landmarks_count = self.state.get_landmark_count()
        return data

    def _normalize_landmarks(self, pts: np.ndarray, w: int, h: int) -> np.ndarray:
        normalized = pts.copy()
        normalized[:, 0] = pts[:, 0] * w
        normalized[:, 1] = pts[:, 1] * h
        return normalized

    def _update_head_pose(self, pose_pts: np.ndarray):
        if pose_pts.shape[0] < 11:
            return
        nose = pose_pts[0]
        left_ear = pose_pts[7]
        right_ear = pose_pts[8]
        left_shoulder = pose_pts[11]
        right_shoulder = pose_pts[12]

        head_center_x = (left_ear[0] + right_ear[0]) / 2.0
        head_center_y = (left_ear[1] + right_ear[1]) / 2.0

        dx = nose[0] - head_center_x
        dy = nose[1] - head_center_y

        self.state.head_pose['x'] = dx
        self.state.head_pose['y'] = dy
        self.state.head_pose['z'] = nose[2] if nose.shape[0] > 2 else 0.0

        shoulder_width = np.linalg.norm(np.array([right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1]]))
        if shoulder_width > 0:
            roll = np.degrees(np.arctan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0]))
            self.state.head_pose['roll'] = roll * self.config.intensity
            yaw = (dx / shoulder_width) * 45.0
            self.state.head_pose['yaw'] = yaw * self.config.intensity
            pitch = (dy / shoulder_width) * 30.0
            self.state.head_pose['pitch'] = pitch * self.config.intensity

    def _update_hand_position(self, side: str, hand_pts: np.ndarray):
        wrist = hand_pts[0]
        index_tip = hand_pts[8]
        center_x = (wrist[0] + index_tip[0]) / 2.0
        center_y = (wrist[1] + index_tip[1]) / 2.0
        old_x, old_y = self.state.hand_positions[side]
        alpha = self.config.smoothing_alpha * 0.5
        smoothed_x = old_x * (1 - alpha) + center_x * alpha
        smoothed_y = old_y * (1 - alpha) + center_y * alpha
        self.state.hand_positions[side] = (smoothed_x, smoothed_y)

    def _detect_blink(self, face_pts: np.ndarray):
        if face_pts.shape[0] < 478:
            return
        left_eye_top = face_pts[159]
        left_eye_bottom = face_pts[145]
        left_eye_left = face_pts[33]
        left_eye_right = face_pts[133]

        right_eye_top = face_pts[386]
        right_eye_bottom = face_pts[374]
        right_eye_left = face_pts[362]
        right_eye_right = face_pts[263]

        ear_left = (np.linalg.norm(left_eye_top - left_eye_bottom)) / (np.linalg.norm(left_eye_left - left_eye_right) + 1e-6)
        ear_right = (np.linalg.norm(right_eye_top - right_eye_bottom)) / (np.linalg.norm(right_eye_left - right_eye_right) + 1e-6)
        ear = (ear_left + ear_right) / 2.0
        self.state.ear_value = ear

        now = time.time()
        if ear < EAR_THRESHOLD and (now - self.last_blink_time) > BLINK_COOLDOWN:
            self.state.blink_state = 1.0
            self.last_blink_time = now

        self.state.blink_state *= 0.85

    def _update_micro_expressions(self):
        self.micro_expression_phase += 0.02
        self.state.micro_expression_offset = np.sin(self.micro_expression_phase) * 0.003 * self.config.intensity
        self.state.head_bob_phase += 0.01

    def _temporal_smoothing(self, data: LandmarkData):
        alpha = self.config.smoothing_alpha
        for key in ['pose', 'left_hand', 'right_hand', 'face_mesh']:
            raw = getattr(data, key)
            if raw is None:
                continue
            if key in self.state.smoothed_landmarks and self.state.smoothed_landmarks[key] is not None:
                prev = self.state.smoothed_landmarks[key]
                if prev.shape == raw.shape:
                    smoothed = prev * alpha + raw * (1 - alpha)
                    self.state.smoothed_landmarks[key] = smoothed
                else:
                    self.state.smoothed_landmarks[key] = raw.copy()
            else:
                self.state.smoothed_landmarks[key] = raw.copy()

    def _apply_physics(self):
        pose_data = self.state.raw_landmarks.get('pose')
        if pose_data is None:
            return

        if 'pose' not in self.state.velocities:
            self.state.velocities['pose'] = np.zeros_like(pose_data)

        if 'pose' in self.state.smoothed_landmarks and self.state.smoothed_landmarks['pose'] is not None:
            prev = self.state.smoothed_landmarks['pose']
            if prev.shape == pose_data.shape:
                vel = (pose_data - prev) * (1.0 / max(self.state.fps, 1))
                self.state.velocities['pose'] = self.state.velocities['pose'] * self.config.friction + vel * (1 - self.config.friction)

        head_tilt_x = self.state.head_pose['yaw'] / 45.0
        head_tilt_y = self.state.head_pose['pitch'] / 30.0

        k = self.config.spring_stiffness
        d = self.config.damping
        m = self.config.mass

        for spring_key, drive_val in [('head_tilt_x', head_tilt_x), ('head_tilt_y', head_tilt_y), ('head_bob', np.sin(self.state.head_bob_phase) * 0.1)]:
            pos = self.state.spring_positions[spring_key]
            vel = self.state.spring_velocities[spring_key]
            force = k * (drive_val - pos) - d * vel
            accel = force / m
            vel += accel
            vel *= self.config.friction
            pos += vel
            self.state.spring_positions[spring_key] = pos
            self.state.spring_velocities[spring_key] = vel

    def get_landmark_data_for_overlay(self) -> Dict[str, Any]:
        return {
            'head_pose': self.state.head_pose.copy(),
            'hand_positions': {k: v for k, v in self.state.hand_positions.items()},
            'blink_state': float(self.state.blink_state),
            'micro_expression': float(self.state.micro_expression_offset),
            'spring': {k: float(v) for k, v in self.state.spring_positions.items()},
            'ear_value': float(self.state.ear_value),
            'landmark_count': self.state.get_landmark_count(),
            'fps': round(self.state.fps, 1),
        }

    def get_config(self) -> dict:
        return {
            'mass': self.config.mass,
            'damping': self.config.damping,
            'spring_stiffness': self.config.spring_stiffness,
            'friction': self.config.friction,
            'gravity': self.config.gravity,
            'smoothing_alpha': self.config.smoothing_alpha,
            'intensity': self.config.intensity,
            'enabled': self.config.enabled,
        }

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

    def close(self):
        self.holistic.close()


_tracker_instance = None


def get_tracker() -> HolisticTracker:
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = HolisticTracker()
    return _tracker_instance