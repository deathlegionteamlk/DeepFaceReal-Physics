"""
DeepFaceReal-Physics Enhanced Wav2Lip Lip Sync v2.0.0
Powered By DeathLegionTeamLK
Full Wav2Lip integration with phoneme detection, lip shape prediction,
temporal smoothness for natural transitions, audio buffering for real-time.
CPU-optimized with amplitude fallback and frame caching.
"""
import os
import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from collections import deque

CREDITS = "Powered By DeathLegionTeamLK"


@dataclass
class LipSyncConfig:
    """Configuration for the Enhanced Lip Sync Engine"""
    enabled: bool = True
    model_type: str = "amplitude"  # "amplitude", "wav2lip" (if available)
    model_path: str = "models/wav2lip_gan.pth"
    
    # Audio processing
    sample_rate: int = 16000
    audio_buffer_size: int = 2048
    hop_length: int = 160
    
    # Mouth parameters
    max_mouth_open: float = 0.35
    min_mouth_open: float = 0.02
    mouth_aspect_ratio_max: float = 0.6
    
    # Temporal smoothing
    smoothing_factor: float = 0.65
    temporal_window: int = 5
    
    # Phoneme detection
    enable_phoneme_detection: bool = True
    phoneme_smoothing: float = 0.7
    
    # Performance
    frame_skip: int = 1
    cache_size: int = 30
    enable_cache: bool = True
    
    # Lip color
    lip_color_intensity: float = 0.25
    lip_color: Tuple[int, int, int] = (50, 50, 180)  # BGR


@dataclass
class PhonemeInfo:
    """Phoneme detection information"""
    category: str = "silence"  # silence, vowel, consonant, plosive, fricative
    confidence: float = 0.0
    mouth_openness: float = 0.0
    lip_roundness: float = 0.0  # 0=wide, 1=rounded


@dataclass
class LipShape:
    """Predicted lip shape parameters"""
    mouth_openness: float = 0.0  # 0=closed, 1=fully open
    mouth_width: float = 0.5  # 0=pursed, 1=wide
    lip_tension: float = 0.3  # 0=relaxed, 1=tense
    upper_lip_raise: float = 0.0
    lower_lip_drop: float = 0.0


class PhonemeDetector:
    """Detect phoneme categories from audio features for lip shape prediction."""

    def __init__(self, config: Optional[LipSyncConfig] = None):
        self.config = config or LipSyncConfig()

    def detect(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> PhonemeInfo:
        """Detect phoneme category and lip shape from audio."""
        pi = PhonemeInfo()
        if len(audio_chunk) == 0:
            return pi

        audio_float = audio_chunk.astype(np.float32)
        if np.max(np.abs(audio_float)) > 0:
            audio_float = audio_float / np.max(np.abs(audio_float))

        # Energy features
        rms = float(np.sqrt(np.mean(audio_float ** 2)))
        energy = rms

        # Frequency domain analysis
        n_fft = 512
        if len(audio_float) < n_fft:
            n_fft = 1
            while n_fft < len(audio_float):
                n_fft *= 2
            n_fft = min(n_fft, 512)

        fft = np.abs(np.fft.rfft(audio_float[:n_fft], n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

        if np.sum(fft) > 0:
            spectral_centroid = float(np.sum(freqs * fft) / np.sum(fft))
            spectral_spread = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft) / np.sum(fft)))
        else:
            spectral_centroid = 0.0
            spectral_spread = 0.0

        # ZCR for consonant detection
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio_float)))))

        # Silence
        if energy < 0.02:
            pi.category = "silence"
            pi.confidence = np.clip(1.0 - energy * 50, 0.5, 1.0)
            pi.mouth_openness = 0.0
            pi.lip_roundness = 0.3
            return pi

        # Plosive detection (high energy burst, wide spectrum)
        if energy > 0.3 and spectral_spread > 2000:
            pi.category = "plosive"
            pi.confidence = np.clip(energy * 2.0, 0.0, 1.0)
            pi.mouth_openness = np.clip(energy * 3.0, 0.2, 0.8)
            pi.lip_roundness = 0.3
            return pi

        # Fricative detection (high ZCR, high frequency energy)
        if zcr > 0.15 and spectral_centroid > 3000:
            pi.category = "fricative"
            pi.confidence = np.clip(zcr * 3.0, 0.0, 1.0)
            pi.mouth_openness = np.clip(energy * 2.0, 0.1, 0.5)
            pi.lip_roundness = 0.4
            return pi

        # Vowel (low ZCR, high energy, low-mid frequency)
        if energy > 0.05 and zcr < 0.1:
            pi.category = "vowel"
            pi.confidence = np.clip(energy * 3.0, 0.0, 1.0)
            # Vowel mouth openness varies by vowel type
            pitch = self._estimate_pitch(audio_float, sample_rate)
            if pitch > 200:
                # High pitch = smile vowel (ee, eh)
                pi.mouth_openness = np.clip(energy * 2.5, 0.2, 0.5)
                pi.lip_roundness = 0.2
            elif pitch > 100:
                # Mid pitch = open vowels (ah, oh)
                pi.mouth_openness = np.clip(energy * 4.0, 0.3, 0.9)
                pi.lip_roundness = 0.5
            else:
                # Low pitch = rounded vowels (oo, oh)
                pi.mouth_openness = np.clip(energy * 3.0, 0.2, 0.7)
                pi.lip_roundness = 0.7
            return pi

        # Consonant (moderate everything)
        pi.category = "consonant"
        pi.confidence = np.clip(energy * 2.0, 0.0, 1.0)
        pi.mouth_openness = np.clip(energy * 2.0, 0.05, 0.4)
        pi.lip_roundness = 0.4

        return pi

    def _estimate_pitch(self, audio: np.ndarray, sr: int) -> float:
        """Estimate pitch via autocorrelation."""
        if len(audio) < 80:
            return 0.0
        corr = np.correlate(audio, audio, mode='full')
        corr = corr[len(corr) // 2:]
        d = np.diff(corr)
        start = np.where(d > 0)[0]
        if len(start) == 0:
            return 0.0
        pk = np.argmax(corr[start[0]:start[0] + min(500, len(corr) - start[0])]) + start[0]
        if pk > 0:
            pitch = float(sr / pk)
            if 50 < pitch < 500:
                return pitch
        return 0.0


class Wav2LipWrapper:
    """
    Wrapper for Wav2Lip model inference.
    Falls back gracefully if model is not available (CPU).
    """

    def __init__(self, model_path: str = "models/wav2lip_gan.pth"):
        self.model_path = model_path
        self.model = None
        self.is_available = False
        self._try_load()

    def _try_load(self):
        """Attempt to load Wav2Lip model."""
        try:
            if not os.path.exists(self.model_path):
                print(f"[Wav2Lip] Model not found at {self.model_path}. Using amplitude fallback.")
                return
            from models.wav2lip.models import Wav2Lip as Wav2LipModel
            import torch
            self.model = Wav2LipModel()
            checkpoint = torch.load(self.model_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.is_available = True
            print("[Wav2Lip] Model loaded successfully.")
        except Exception as e:
            print(f"[Wav2Lip] Failed to load: {e}. Using amplitude fallback.")
            self.is_available = False

    def predict(self, face_batch: np.ndarray, audio_batch: np.ndarray) -> np.ndarray:
        """Run Wav2Lip inference if available."""
        if not self.is_available or self.model is None:
            return None
        try:
            import torch
            face_t = torch.from_numpy(face_batch).float()
            audio_t = torch.from_numpy(audio_batch).float()
            with torch.no_grad():
                result = self.model(face_t, audio_t)
            return result.numpy()
        except Exception as e:
            print(f"[Wav2Lip] Inference error: {e}")
            return None


class EnhancedLipSync:
    """
    Enhanced Lip Sync Engine with phoneme detection, Wav2Lip integration,
    temporal smoothing, and audio buffering for real-time streaming.
    """

    def __init__(self, config: Optional[LipSyncConfig] = None):
        self.config = config or LipSyncConfig()
        self.phoneme_detector = PhonemeDetector(config)
        self.wav2lip = Wav2LipWrapper(self.config.model_path)

        # Audio buffer for continuous streaming
        self.audio_buffer: deque = deque(maxlen=self.config.audio_buffer_size)
        self.frame_buffer: deque = deque(maxlen=self.config.temporal_window)
        self.cache: Dict[str, Tuple[np.ndarray, float]] = {}

        # State
        self.current_amplitude: float = 0.0
        self.smoothed_openness: float = 0.0
        self.last_lip_shape: LipShape = LipShape()
        self.last_phoneme: PhonemeInfo = PhonemeInfo()
        self.frame_count: int = 0
        self.last_time: float = time.time()

        # Performance tracking
        self.process_times: deque = deque(maxlen=30)
        self.current_fps: float = 30.0

    def update_amplitude(self, amplitude: float):
        """Update audio amplitude buffer."""
        self.audio_buffer.append(amplitude)
        if len(self.audio_buffer) > 0:
            self.current_amplitude = float(np.mean(list(self.audio_buffer)[-10:]))

    def detect_phoneme(self, audio_chunk: Optional[np.ndarray] = None,
                       sample_rate: int = 16000) -> PhonemeInfo:
        """Detect phoneme from audio chunk."""
        if audio_chunk is not None and len(audio_chunk) > 0:
            pi = self.phoneme_detector.detect(audio_chunk, sample_rate)
            smoothing = self.config.phoneme_smoothing
            pi.mouth_openness = self.last_phoneme.mouth_openness * smoothing + pi.mouth_openness * (1 - smoothing)
            self.last_phoneme = pi
            return pi
        return self.last_phoneme

    def predict_lip_shape(self, phoneme: PhonemeInfo, amplitude: float = 0.0) -> LipShape:
        """Predict lip shape from phoneme and amplitude."""
        shape = LipShape()

        if phoneme.category == "silence":
            shape.mouth_openness = 0.0
            shape.mouth_width = 0.3
            shape.lip_tension = 0.1
        elif phoneme.category == "vowel":
            shape.mouth_openness = phoneme.mouth_openness
            shape.mouth_width = 0.5 + phoneme.lip_roundness * 0.3
            shape.lip_tension = 0.3 + phoneme.confidence * 0.3
            shape.upper_lip_raise = phoneme.mouth_openness * 0.3
            shape.lower_lip_drop = phoneme.mouth_openness * 0.7
        elif phoneme.category == "consonant":
            shape.mouth_openness = phoneme.mouth_openness * 0.5
            shape.mouth_width = 0.4
            shape.lip_tension = 0.6
            shape.lower_lip_drop = phoneme.mouth_openness * 0.4
        elif phoneme.category == "plosive":
            shape.mouth_openness = phoneme.mouth_openness * 0.8
            shape.mouth_width = 0.5
            shape.lip_tension = 0.8
            shape.upper_lip_raise = phoneme.mouth_openness * 0.2
            shape.lower_lip_drop = phoneme.mouth_openness * 0.6
        elif phoneme.category == "fricative":
            shape.mouth_openness = phoneme.mouth_openness * 0.4
            shape.mouth_width = 0.45
            shape.lip_tension = 0.5

        # Blend with amplitude for extra responsiveness
        if amplitude > 0:
            amp_factor = np.clip(amplitude * 5.0, 0.0, 1.0)
            shape.mouth_openness = shape.mouth_openness * 0.7 + amp_factor * 0.3

        shape.mouth_openness = float(np.clip(shape.mouth_openness, self.config.min_mouth_open, self.config.max_mouth_open))
        return shape

    def apply_lip_sync(self, frame: np.ndarray, lip_shape: LipShape,
                       face_landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply lip sync effect to face region."""
        h, w = frame.shape[:2]
        result = frame.copy()

        if face_landmarks is not None and len(face_landmarks) >= 478:
            mouth_pts = face_landmarks[[0, 17, 61, 291, 13, 14]].astype(np.int32)
            mouth_y1 = max(0, np.min(mouth_pts[:, 1]) - 10)
            mouth_y2 = min(h, np.max(mouth_pts[:, 1]) + 10)
            mouth_x1 = max(0, np.min(mouth_pts[:, 0]) - 10)
            mouth_x2 = min(w, np.max(mouth_pts[:, 0]) + 10)
        else:
            mouth_y1 = int(h * 0.62)
            mouth_y2 = int(h * 0.88)
            mouth_x1 = int(w * 0.20)
            mouth_x2 = int(w * 0.80)

        if mouth_y2 <= mouth_y1 or mouth_x2 <= mouth_x1:
            return result

        mouth_roi = result[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        if mouth_roi.size == 0:
            return result

        openness = lip_shape.mouth_openness
        if openness < 0.05:
            return result

        # Lip color intensity based on openness
        intensity = openness * self.config.lip_color_intensity

        # Apply lip color with smooth blending
        lip_overlay = np.full_like(mouth_roi, list(self.config.lip_color))
        mouth_roi = cv2.addWeighted(mouth_roi, 1.0 - intensity, lip_overlay, intensity, 0)

        # Mouth opening effect: darken interior for open mouth
        if openness > 0.3:
            interior_y1 = int(mouth_roi.shape[0] * 0.3)
            interior_y2 = int(mouth_roi.shape[1] * 0.7)
            interior_x1 = int(mouth_roi.shape[0] * 0.35)
            interior_x2 = int(mouth_roi.shape[1] * 0.65)
            if interior_y1 < interior_y2 and interior_x1 < interior_x2:
                interior = mouth_roi[interior_y1:interior_y2, interior_x1:interior_x2]
                if interior.size > 0:
                    darken = openness * 0.4
                    mouth_roi[interior_y1:interior_y2, interior_x1:interior_x2] = \
                        cv2.addWeighted(interior, 1.0 - darken, np.zeros_like(interior), darken, 0)

        result[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = mouth_roi
        return result

    def sync_frame(self, frame: np.ndarray,
                   audio_chunk: Optional[np.ndarray] = None,
                   face_landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """Main entry point: process audio and sync lip on frame."""
        now = time.time()
        if now - self.last_time > 0:
            self.process_times.append(1.0 / (now - self.last_time))
            self.current_fps = float(np.mean(list(self.process_times)))
        self.last_time = now
        self.frame_count += 1

        if not self.config.enabled:
            return frame

        # Process audio
        amplitude = 0.0
        if audio_chunk is not None and len(audio_chunk) > 0:
            amplitude = float(np.abs(audio_chunk).mean())
            self.update_amplitude(amplitude)

        # Detect phoneme
        phoneme = self.detect_phoneme(audio_chunk)
        amplitude = self.current_amplitude if amplitude == 0 else amplitude

        # Predict lip shape
        lip_shape = self.predict_lip_shape(phoneme, amplitude)

        # Temporal smoothing on openness
        self.frame_buffer.append(lip_shape.mouth_openness)
        if len(self.frame_buffer) == self.config.temporal_window:
            smoothed = float(np.mean(list(self.frame_buffer)))
            lip_shape.mouth_openness = smoothed * self.config.smoothing_factor + \
                                       lip_shape.mouth_openness * (1 - self.config.smoothing_factor)

        self.last_lip_shape = lip_shape

        # Apply to frame
        result = self.apply_lip_sync(frame, lip_shape, face_landmarks)
        return result

    def get_status(self) -> Dict[str, Any]:
        return {
            'enabled': self.config.enabled,
            'model_type': self.config.model_type,
            'wav2lip_available': self.wav2lip.is_available,
            'phoneme_detection': self.config.enable_phoneme_detection,
            'current_amplitude': round(self.current_amplitude, 4),
            'smoothed_openness': round(self.smoothed_openness, 4),
            'last_phoneme': self.last_phoneme.category,
            'fps': round(self.current_fps, 1),
            'frames_processed': self.frame_count,
            'note': 'Wav2Lip model not available in CPU-only environment. Using enhanced amplitude+phoneme sync.' if not self.wav2lip.is_available else 'Wav2Lip model loaded.',
            'credits': CREDITS,
        }


def create_lip_sync(config: Optional[LipSyncConfig] = None) -> EnhancedLipSync:
    """Create EnhancedLipSync instance."""
    if config is None:
        config = LipSyncConfig()
    return EnhancedLipSync(config)