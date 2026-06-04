
import os
import cv2
import numpy as np
from typing import Optional, List, Any
from dataclasses import dataclass

CREDITS = 'Powered By DeathLegionTeamLK'

@dataclass
class LipSyncConfig:
    enabled: bool = False
    model_type: str = 'amplitude'
    audio_buffer_size: int = 512
    max_mouth_open: float = 0.35
    min_mouth_open: float = 0.02
    smoothing_factor: float = 0.7

class AmplitudeLipSync:

    def __init__(self, config: LipSyncConfig = None):
        self.config = config or LipSyncConfig()
        self.audio_buffer: List[float] = []
        self.current_amplitude: float = 0.0
        self.smoothed_openness: float = 0.0

    def update_amplitude(self, amplitude: float):
        self.audio_buffer.append(amplitude)
        if len(self.audio_buffer) > self.config.audio_buffer_size:
            self.audio_buffer.pop(0)

        if len(self.audio_buffer) > 0:
            self.current_amplitude = np.mean(self.audio_buffer[-10:])

    def get_mouth_openness(self) -> float:
        raw_openness = np.clip(
            self.current_amplitude * 3.0,
            self.config.min_mouth_open,
            self.config.max_mouth_open
        )

        self.smoothed_openness = (
            self.smoothed_openness * self.config.smoothing_factor
            + raw_openness * (1 - self.config.smoothing_factor)
        )

        return min(self.smoothed_openness / self.config.max_mouth_open, 1.0)

    def apply_lip_sync(self, face_roi: np.ndarray, mouth_openness: float) -> np.ndarray:
        if mouth_openness < 0.1:
            return face_roi

        h, w = face_roi.shape[:2]
        result = face_roi.copy()

        mouth_y1 = int(h * 0.65)
        mouth_y2 = int(h * 0.85)
        mouth_x1 = int(w * 0.25)
        mouth_x2 = int(w * 0.75)

        mouth_roi = result[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        if mouth_roi.size == 0:
            return result

        lip_color = np.array([50, 50, 180], dtype=np.uint8)
        intensity = mouth_openness * 0.3

        mouth_roi = cv2.addWeighted(
            mouth_roi, 1.0 - intensity,
            np.full_like(mouth_roi, lip_color), intensity,
            0
        )

        result[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = mouth_roi
        return result

    def sync_frame(self, frame: np.ndarray, audio_chunk: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.config.enabled:
            return frame

        if audio_chunk is not None:
            amplitude = np.abs(audio_chunk).mean()
            self.update_amplitude(amplitude)

        openness = self.get_mouth_openness()
        return self.apply_lip_sync(frame, openness)

    def get_status(self) -> dict:
        return {
            'enabled': self.config.enabled,
            'model_type': 'amplitude',
            'note': 'Wav2Lip not available in CPU-only environment. Using amplitude-based sync.'
        }

def create_lip_sync(config: Optional[LipSyncConfig] = None) -> AmplitudeLipSync:
    if config is None:
        config = LipSyncConfig()
    return AmplitudeLipSync(config)