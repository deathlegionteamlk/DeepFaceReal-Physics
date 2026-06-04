import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from typing import Optional, List, Tuple, Dict, Any

CREDITS = 'Powered By DeathLegionTeamLK'

from .physics_engine import get_tracker, HolisticTracker
from .background_engine import get_background, ParallaxBackground


class FaceSwapper:

    def __init__(self, det_size: Tuple[int, int] = (320, 320), providers: Optional[List[str]] = None):
        if providers is None:
            providers = ['CPUExecutionProvider']
        print(f"[FaceSwapper] Initializing with det_size={det_size}...")
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)
        print(f"[FaceSwapper] FaceAnalysis loaded: {len(self.app.models)} models")
        model_path = os.path.expanduser('~/.insightface/models/inswapper_128.onnx')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"inswapper_128.onnx not found at {model_path}")
        self.swapper = get_model(model_path)
        print(f"[FaceSwapper] INSwapper loaded from {model_path}")
        self.source_faces: Dict[str, Dict[str, Any]] = {}
        self._sharpen_kernel = np.array([
            [0, -1, 0], [-1, 5, -1], [0, -1, 0]
        ], dtype=np.float32)
        self.tracker = get_tracker()
        self.background = get_background()
        self.physics_enabled = True
        self.hand_overlay_enabled = True
        self.background_mode = 'parallax'
        self.hand_skeleton_color = (0, 255, 0)
        self.hand_skeleton_thickness = 2
        self._last_face_center = None
        self._smooth_face_center = None
        self._frame_buffer_for_physics = 0

    def get_source_face(self, img: np.ndarray) -> Any:
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f.det_score)

    def extract_face_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        face = self.get_source_face(img)
        if face is None:
            return None
        return face.embedding

    def detect_faces(self, img: np.ndarray) -> List[Any]:
        return self.app.get(img)

    def swap_face(self, img: np.ndarray, source_face: Any, target_faces: Optional[List[Any]] = None, blend: bool = True, enhance: bool = True, enhance_quality: str = 'medium') -> np.ndarray:
        if target_faces is None:
            target_faces = self.detect_faces(img)
        if len(target_faces) == 0:
            return img
        result = img.copy()
        for target_face in target_faces:
            swapped = self.swapper.get(result, target_face, source_face, paste_back=not blend)
            if blend:
                result = self._poisson_blend(result, swapped, target_face)
            else:
                result = swapped
            if enhance:
                result = self.enhance_face(result, quality=enhance_quality)
        return result

    def _poisson_blend(self, original: np.ndarray, swapped: np.ndarray, face: Any) -> np.ndarray:
        h, w = original.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        bbox = face.bbox.astype(np.int32)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 11)
        try:
            result = cv2.seamlessClone(swapped, original, mask, center, cv2.NORMAL_CLONE)
            return result
        except Exception as e:
            print(f"[FaceSwapper] Poisson blend failed ({e}), using direct paste")
            return swapped

    def enhance_face(self, img: np.ndarray, quality: str = 'medium') -> np.ndarray:
        if quality == 'low':
            return img
        result = img.copy()
        if quality in ('medium', 'high'):
            result = cv2.bilateralFilter(result, 5, 30, 30)
        sharpen_strength = 1.0
        if quality == 'high':
            sharpen_strength = 1.5
        sharpened = cv2.filter2D(result, -1, self._sharpen_kernel * sharpen_strength)
        result = cv2.addWeighted(result, 0.4, sharpened, 0.6, 0)
        if quality == 'high':
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return result

    def _draw_hand_skeleton(self, img: np.ndarray, landmarks: np.ndarray, color: Tuple[int, int, int], thickness: int = 2):
        if landmarks is None or len(landmarks) < 21:
            return
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20)
        ]
        pts_2d = landmarks[:, :2].astype(np.int32)
        for start, end in connections:
            if start < len(pts_2d) and end < len(pts_2d):
                pt1 = tuple(pts_2d[start])
                pt2 = tuple(pts_2d[end])
                cv2.line(img, pt1, pt2, color, thickness)
        for pt in pts_2d:
            cv2.circle(img, tuple(pt), 3, (255, 255, 255), -1)

    def _draw_face_mesh_outline(self, img: np.ndarray, landmarks: np.ndarray):
        if landmarks is None or len(landmarks) < 468:
            return
        pts_2d = landmarks[:, :2].astype(np.int32)
        jaw = list(range(0, 17))
        left_eyebrow = list(range(17, 22))
        right_eyebrow = list(range(22, 27))
        nose_bridge = list(range(27, 31))
        nose_bottom = list(range(31, 36))
        left_eye = list(range(36, 42))
        right_eye = list(range(42, 48))
        outer_lip = list(range(61, 68))
        for idx_set, color in [(jaw, (200, 200, 200)), (left_eyebrow, (180, 180, 100)), (right_eyebrow, (180, 180, 100)),
                                (nose_bridge, (100, 200, 100)), (nose_bottom, (100, 200, 100)),
                                (left_eye, (200, 200, 100)), (right_eye, (200, 200, 100)),
                                (outer_lip, (100, 100, 200))]:
            for i in range(len(idx_set) - 1):
                cv2.line(img, tuple(pts_2d[idx_set[i]]), tuple(pts_2d[idx_set[i+1]]), color, 1)

    def _apply_blink_effect(self, img: np.ndarray, blink_state: float) -> np.ndarray:
        if blink_state < 0.05:
            return img
        h, w = img.shape[:2]
        overlay = img.copy()
        alpha = blink_state * 0.6
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        result = cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0)
        return result

    def _smooth_face_center(self, detected_center: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if detected_center is None:
            self._smooth_face_center = None
            return None
        if self._smooth_face_center is None:
            self._smooth_face_center = detected_center
            return detected_center
        alpha = 0.4
        sx = self._smooth_face_center[0] * (1 - alpha) + detected_center[0] * alpha
        sy = self._smooth_face_center[1] * (1 - alpha) + detected_center[1] * alpha
        self._smooth_face_center = (sx, sy)
        return self._smooth_face_center

    def process_frame(self, frame: np.ndarray, source_face: Any, blend: bool = True, enhance: bool = True, enhance_quality: str = 'medium', min_face_size: int = 50) -> np.ndarray:
        faces = self.detect_faces(frame)
        if len(faces) == 0:
            if self.background_mode in ('parallax', 'blur'):
                head_x = getattr(self.tracker.state.head_pose, 'get', lambda k: 0.0)('x') if hasattr(self.tracker, 'state') else 0.0
                head_y = getattr(self.tracker.state.head_pose, 'get', lambda k: 0.0)('y') if hasattr(self.tracker, 'state') else 0.0
                bg = self.background.render(head_x, head_y)
                if bg.shape[:2] == frame.shape[:2]:
                    return bg
            return frame

        valid_faces = []
        for face in faces:
            bbox = face.bbox.astype(np.int32)
            fw = bbox[2] - bbox[0]
            fh = bbox[3] - bbox[1]
            if fw >= min_face_size and fh >= min_face_size:
                valid_faces.append(face)

        if len(valid_faces) == 0:
            return frame

        landmark_data = self.tracker.process_frame(frame)
        overlay_data = self.tracker.get_landmark_data_for_overlay()
        self._frame_buffer_for_physics += 1

        head_x = self.tracker.state.head_pose.get('x', 0.0)
        head_y = self.tracker.state.head_pose.get('y', 0.0)

        if self.physics_enabled:
            inertia_x = self.tracker.state.spring_positions.get('head_tilt_x', 0.0) * 20.0
            inertia_y = self.tracker.state.spring_positions.get('head_tilt_y', 0.0) * 20.0
        else:
            inertia_x = 0.0
            inertia_y = 0.0

        result = frame.copy()

        if self.background_mode in ('parallax', 'blur'):
            self.background.config.mode = self.background_mode
            bg = self.background.render(head_x + inertia_x, head_y + inertia_y)
            if bg.shape[:2] == frame.shape[:2]:
                result = bg

        source_face_use = source_face
        target_face = valid_faces[0]
        face_bbox = target_face.bbox.astype(np.int32)
        face_center_x = (face_bbox[0] + face_bbox[2]) // 2
        face_center_y = (face_bbox[1] + face_bbox[3]) // 2

        if self.physics_enabled:
            smoothed = self._smooth_face_center((float(face_center_x), float(face_center_y)))
            if smoothed:
                face_center_x = int(smoothed[0])
                face_center_y = int(smoothed[1])

        swapped = self.swapper.get(result, target_face, source_face_use, paste_back=False)
        if blend:
            h, w = result.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]
            pad_x = int((x2 - x1) * 0.15)
            pad_y = int((y2 - y1) * 0.15)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 11)
            try:
                result = cv2.seamlessClone(swapped, result, mask, center, cv2.NORMAL_CLONE)
            except Exception:
                result = swapped
        else:
            result = swapped

        if enhance:
            result = self.enhance_face(result, quality=enhance_quality)

        blink = overlay_data.get('blink_state', 0.0)
        if blink > 0.05:
            result = self._apply_blink_effect(result, blink)

        if self.hand_overlay_enabled:
            left_hand = landmark_data.left_hand if landmark_data is not None else None
            right_hand = landmark_data.right_hand if landmark_data is not None else None
            if left_hand is not None:
                self._draw_hand_skeleton(result, left_hand, (0, 200, 0), self.hand_skeleton_thickness)
            if right_hand is not None:
                self._draw_hand_skeleton(result, right_hand, (0, 200, 200), self.hand_skeleton_thickness)

        face_mesh = landmark_data.face_mesh if landmark_data is not None else None
        if face_mesh is not None and self.hand_overlay_enabled:
            pass

        return result

    def get_face_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        face = self.get_source_face(img)
        if face is None:
            return None
        return face.normed_embedding

    def compare_faces(self, emb1: np.ndarray, emb2: np.ndarray, threshold: float = 0.5):
        sim = np.dot(emb1, emb2)
        return sim > threshold, sim

    def get_upscaled_face(self, img: np.ndarray, source_face: Any, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        bbox = source_face.bbox.astype(np.int32)
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])
        face_roi = img[y1:y2, x1:x2]
        if face_roi.size == 0:
            return None
        return cv2.resize(face_roi, target_size, interpolation=cv2.INTER_CUBIC)


_swapper_instance: Optional[FaceSwapper] = None


def get_swapper(det_size: Tuple[int, int] = (320, 320)) -> FaceSwapper:
    global _swapper_instance
    if _swapper_instance is None:
        _swapper_instance = FaceSwapper(det_size=det_size)
    return _swapper_instance