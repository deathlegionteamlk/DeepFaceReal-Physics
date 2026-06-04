"""
DeepFaceReal-Physics 3D Face Engine v2.0.0
Powered By DeathLegionTeamLK
3D-aware face reconstruction using MediaPipe face mesh (468 landmarks)
Head pose estimation (6DoF), expression coefficients, Delaunay triangulation for 3D mesh
"""
import cv2
import numpy as np
import mediapipe as mp
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from collections import deque

CREDITS = "Powered By DeathLegionTeamLK"

mp_face_mesh = mp.solutions.face_mesh


@dataclass
class Face3DConfig:
    """Configuration for the 3D Face Engine"""
    static_image_mode: bool = False
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    frame_skip: int = 2
    smoothing_alpha: float = 0.65
    enable_expression_extraction: bool = True
    enable_pose_estimation: bool = True
    enable_mesh_generation: bool = True
    max_triangles: int = 900
    mesh_scale: float = 1.0


@dataclass
class HeadPose6DoF:
    """6 Degrees of Freedom head pose"""
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {'pitch': self.pitch, 'yaw': self.yaw, 'roll': self.roll, 'tx': self.tx, 'ty': self.ty, 'tz': self.tz}

    def to_array(self) -> np.ndarray:
        return np.array([self.pitch, self.yaw, self.roll, self.tx, self.ty, self.tz])


@dataclass
class ExpressionCoefficients:
    """Facial expression coefficients (blendshape-like)"""
    brow_raise_left: float = 0.0
    brow_raise_right: float = 0.0
    brow_furrow: float = 0.0
    eye_open_left: float = 1.0
    eye_open_right: float = 1.0
    nose_wrinkle: float = 0.0
    mouth_open: float = 0.0
    mouth_smile: float = 0.0
    mouth_pucker: float = 0.0
    jaw_drop: float = 0.0
    cheek_raise: float = 0.0
    lip_press: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'brow_raise_left': self.brow_raise_left, 'brow_raise_right': self.brow_raise_right,
            'brow_furrow': self.brow_furrow, 'eye_open_left': self.eye_open_left,
            'eye_open_right': self.eye_open_right, 'nose_wrinkle': self.nose_wrinkle,
            'mouth_open': self.mouth_open, 'mouth_smile': self.mouth_smile,
            'mouth_pucker': self.mouth_pucker, 'jaw_drop': self.jaw_drop,
            'cheek_raise': self.cheek_raise, 'lip_press': self.lip_press,
        }

    def to_array(self) -> np.ndarray:
        return np.array([
            self.brow_raise_left, self.brow_raise_right, self.brow_furrow,
            self.eye_open_left, self.eye_open_right, self.nose_wrinkle,
            self.mouth_open, self.mouth_smile, self.mouth_pucker,
            self.jaw_drop, self.cheek_raise, self.lip_press
        ])


@dataclass
class Face3DData:
    """Complete 3D face data output"""
    landmarks_2d: np.ndarray = None
    landmarks_3d: np.ndarray = None
    head_pose: HeadPose6DoF = field(default_factory=HeadPose6DoF)
    expressions: ExpressionCoefficients = field(default_factory=ExpressionCoefficients)
    triangles: np.ndarray = None
    mesh_vertices: np.ndarray = None
    mesh_faces: np.ndarray = None
    face_detected: bool = False
    timestamp: float = 0.0
    frame_height: int = 0
    frame_width: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'face_detected': self.face_detected, 'timestamp': self.timestamp,
            'head_pose': self.head_pose.to_dict() if self.head_pose else {},
            'expressions': self.expressions.to_dict() if self.expressions else {},
            'landmarks_count': len(self.landmarks_2d) if self.landmarks_2d is not None else 0,
            'triangles_count': len(self.triangles) if self.triangles is not None else 0,
            'resolution': f"{self.frame_width}x{self.frame_height}",
        }


class Face3DEngine:
    """3D Face Engine using MediaPipe Face Mesh for 468 landmark detection,
    6DoF head pose estimation, expression coefficients, and Delaunay triangulation."""

    def __init__(self, config: Optional[Face3DConfig] = None):
        self.config = config or Face3DConfig()
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=self.config.static_image_mode,
            max_num_faces=self.config.max_num_faces,
            refine_landmarks=self.config.refine_landmarks,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self.frame_count = 0
        self.frame_skip_counter = 0
        self._smoothed_landmarks: Optional[np.ndarray] = None
        self._smoothed_landmarks_3d: Optional[np.ndarray] = None
        self._smoothed_pose: Optional[HeadPose6DoF] = None
        self.fps_history = deque(maxlen=30)
        self.last_process_time = 0.0
        self.current_fps: float = 0.0
        self._cached_data: Optional[Face3DData] = None
        self._delaunay_cache = None
        self._init_reference_points()

    def _init_reference_points(self):
        """Reference 3D model points for PnP pose estimation (MediaPipe canonical)."""
        self.ref_3d_points = np.array([
            [0.0, -3.0, 0.0], [-1.0, -2.5, -1.5], [1.0, -2.5, -1.5],
            [-2.5, -3.0, -2.5], [2.5, -3.0, -2.5],
            [-3.0, 1.0, -2.0], [3.0, 1.0, -2.0],
            [0.0, 3.5, -1.5], [-2.0, -4.5, -3.0], [2.0, -4.5, -3.0],
        ], dtype=np.float64)
        self.ref_indices = [1, 2, 4, 33, 263, 61, 291, 152, 234, 454]

    def process_frame(self, frame: np.ndarray) -> Face3DData:
        """Process frame and extract 3D face data."""
        h, w = frame.shape[:2]
        now = time.time()
        if self.last_process_time > 0:
            dt = now - self.last_process_time
            if dt > 0:
                self.fps_history.append(1.0 / dt)
                self.current_fps = sum(self.fps_history) / max(len(self.fps_history), 1)
        self.last_process_time = now

        self.frame_skip_counter += 1
        if self.frame_skip_counter % (self.config.frame_skip + 1) != 0:
            return self._get_cached_data(h, w, now)

        self.frame_count += 1
        data = Face3DData(timestamp=now, frame_height=h, frame_width=w)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            face_landmarks = results.multi_face_landmarks[0]
            data.face_detected = True
            l2d, l3d = [], []
            for lm in face_landmarks.landmark:
                l2d.append([lm.x * w, lm.y * h])
                l3d.append([lm.x, lm.y, lm.z])
            data.landmarks_2d = np.array(l2d, dtype=np.float32)
            data.landmarks_3d = np.array(l3d, dtype=np.float32)

            if self._smoothed_landmarks is not None and data.landmarks_2d.shape == self._smoothed_landmarks.shape:
                alpha = self.config.smoothing_alpha
                data.landmarks_2d = self._smoothed_landmarks * alpha + data.landmarks_2d * (1.0 - alpha)
                data.landmarks_3d = self._smoothed_landmarks_3d * alpha + data.landmarks_3d * (1.0 - alpha)

            self._smoothed_landmarks = data.landmarks_2d.copy()
            self._smoothed_landmarks_3d = data.landmarks_3d.copy()

            if self.config.enable_pose_estimation:
                data.head_pose = self._estimate_head_pose(data.landmarks_2d, w, h)
            if self.config.enable_expression_extraction:
                data.expressions = self._extract_expressions(data.landmarks_2d)
            if self.config.enable_mesh_generation:
                data.triangles = self._compute_delaunay_mesh(data.landmarks_2d, h, w)
                data.mesh_vertices = data.landmarks_3d * self.config.mesh_scale
                data.mesh_faces = data.triangles
            self._cached_data = data
        return data

    def _get_cached_data(self, h: int, w: int, ts: float) -> Face3DData:
        if self._cached_data is not None:
            d = self._cached_data
            d.timestamp = ts
            d.frame_height = h
            d.frame_width = w
            return d
        return Face3DData(timestamp=ts, frame_height=h, frame_width=w)

    def _estimate_head_pose(self, landmarks_2d: np.ndarray, img_w: int, img_h: int) -> HeadPose6DoF:
        """Estimate 6DoF head pose using PnP."""
        pose = HeadPose6DoF()
        if landmarks_2d is None or len(landmarks_2d) < 468:
            return pose
        img_pts = []
        for idx in self.ref_indices:
            if idx < len(landmarks_2d):
                img_pts.append(landmarks_2d[idx])
        if len(img_pts) < 6:
            return pose
        img_pts = np.array(img_pts, dtype=np.float64)
        focal = img_w
        center = (img_w / 2, img_h / 2)
        cam_mat = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)
        try:
            success, rvec, tvec = cv2.solvePnP(self.ref_3d_points, img_pts, cam_mat, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                R, _ = cv2.Rodrigues(rvec)
                sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
                if sy > 1e-6:
                    pose.roll = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                    pose.pitch = float(np.degrees(np.arctan2(-R[2, 0], sy)))
                    pose.yaw = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
                else:
                    pose.roll = float(np.degrees(np.arctan2(-R[1, 2], R[1, 1])))
                    pose.pitch = float(np.degrees(np.arctan2(-R[2, 0], sy)))
                pose.tx = float(tvec[0])
                pose.ty = float(tvec[1])
                pose.tz = float(tvec[2])
        except Exception:
            pass
        if self._smoothed_pose is not None:
            a = self.config.smoothing_alpha * 0.8
            for attr in ['pitch', 'yaw', 'roll', 'tx', 'ty', 'tz']:
                setattr(pose, attr, getattr(self._smoothed_pose, attr) * a + getattr(pose, attr) * (1.0 - a))
        self._smoothed_pose = pose
        return pose

    def _extract_expressions(self, l2d: np.ndarray) -> ExpressionCoefficients:
        """Extract expression coefficients from landmarks."""
        exp = ExpressionCoefficients()
        if l2d is None or len(l2d) < 468:
            return exp

        def d(i, j):
            return float(np.linalg.norm(l2d[i] - l2d[j]))

        leh, lew = d(159, 145), d(33, 133)
        reh, rew = d(386, 374), d(362, 263)
        if lew > 0:
            exp.eye_open_left = float(np.clip(leh / lew * 3.0, 0.0, 1.0))
        if rew > 0:
            exp.eye_open_right = float(np.clip(reh / rew * 3.0, 0.0, 1.0))

        moh, mw = d(13, 14), d(61, 291)
        if mw > 0:
            mr = (moh / mw) * 2.5
            exp.mouth_open = float(np.clip(mr, 0.0, 1.0))
            exp.jaw_drop = exp.mouth_open

        sy = (l2d[61, 1] + l2d[291, 1]) / 2 - l2d[13, 1]
        exp.mouth_smile = float(np.clip(-sy / (mw * 0.5 + 1e-6), 0.0, 1.0))

        lbh = d(105, 107) / (d(33, 133) + 1e-6)
        rbh = d(334, 336) / (d(362, 263) + 1e-6)
        exp.brow_raise_left = float(np.clip((lbh - 0.3) * 3.0, 0.0, 1.0))
        exp.brow_raise_right = float(np.clip((rbh - 0.3) * 3.0, 0.0, 1.0))
        bd = d(105, 334) / 480.0
        exp.brow_furrow = float(np.clip(1.0 - bd * 5.0, 0.0, 1.0))
        lph = d(0, 17)
        exp.lip_press = float(np.clip(1.0 - d(13, 14) / (lph + 1e-6) * 5.0, 0.0, 1.0))
        lc = d(50, 205) / (d(33, 133) + 1e-6)
        exp.cheek_raise = float(np.clip((lc - 1.5) * 2.0, 0.0, 1.0))
        exp.mouth_pucker = float(np.clip(1.0 - d(61, 291) / (mw + 1e-6), 0.0, 1.0))
        return exp

    def _compute_delaunay_mesh(self, l2d: np.ndarray, h: int, w: int) -> np.ndarray:
        """Compute Delaunay triangulation for 3D surface mapping."""
        if l2d is None or len(l2d) < 3:
            return np.array([])
        ck = len(l2d)
        if self._delaunay_cache is not None and self._delaunay_cache[0] == ck:
            return self._delaunay_cache[1]
        from scipy.spatial import Delaunay
        pts = l2d.astype(np.float64)
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)
        combined = np.vstack([pts, corners])
        try:
            tri = Delaunay(combined)
            tris = tri.simplices
            valid = [t for t in tris if np.all(t < len(l2d))]
            tris = np.array(valid, dtype=np.int32)
            if len(tris) > self.config.max_triangles:
                idx = np.random.choice(len(tris), self.config.max_triangles, replace=False)
                tris = tris[idx]
            self._delaunay_cache = (ck, tris)
            return tris
        except Exception:
            return np.array([])

    def render_mesh_overlay(self, frame: np.ndarray, face_data: Face3DData,
                            show_landmarks: bool = True, show_triangles: bool = False,
                            show_pose: bool = True) -> np.ndarray:
        """Render 3D face mesh overlay for visualization."""
        if not face_data.face_detected or face_data.landmarks_2d is None:
            return frame
        result = frame.copy()
        h, w = frame.shape[:2]
        pts = face_data.landmarks_2d.astype(np.int32)
        if show_triangles and face_data.triangles is not None and len(face_data.triangles) > 0:
            for tri in face_data.triangles[:200]:
                if np.all(tri < len(pts)):
                    cv2.line(result, tuple(pts[tri[0]]), tuple(pts[tri[1]]), (100, 200, 100), 1)
                    cv2.line(result, tuple(pts[tri[1]]), tuple(pts[tri[2]]), (100, 200, 100), 1)
                    cv2.line(result, tuple(pts[tri[2]]), tuple(pts[tri[0]]), (100, 200, 100), 1)
        if show_landmarks:
            for pt in pts:
                cv2.circle(result, tuple(pt), 2, (0, 255, 0), -1)
        if show_pose and face_data.head_pose:
            p = face_data.head_pose
            cv2.putText(result, f"P:{p.pitch:.1f} Y:{p.yaw:.1f} R:{p.roll:.1f}", (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        if face_data.expressions:
            e = face_data.expressions
            cv2.putText(result, f"Mouth:{e.mouth_open:.2f} Smile:{e.mouth_smile:.2f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(result, CREDITS, (w - 250, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        return result

    def get_status(self) -> Dict[str, Any]:
        return {
            'fps': round(self.current_fps, 1), 'frames_processed': self.frame_count,
            'config': {'frame_skip': self.config.frame_skip, 'smoothing_alpha': self.config.smoothing_alpha,
                       'enable_pose': self.config.enable_pose_estimation,
                       'enable_expressions': self.config.enable_expression_extraction,
                       'enable_mesh': self.config.enable_mesh_generation},
            'credits': CREDITS,
        }

    def close(self):
        self.face_mesh.close()


_face_3d_engine_instance: Optional[Face3DEngine] = None


def get_face_3d_engine() -> Face3DEngine:
    global _face_3d_engine_instance
    if _face_3d_engine_instance is None:
        _face_3d_engine_instance = Face3DEngine()
    return _face_3d_engine_instance