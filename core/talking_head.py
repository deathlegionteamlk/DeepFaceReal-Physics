"""
DeepFaceReal-Physics Audio-Driven Talking Head v2.0.0
Powered By DeathLegionTeamLK
Audio feature extraction (MFCC, pitch, energy), head pose prediction from audio,
expression generation synchronized to speech, natural head motion patterns.
Inspired by SadTalker/Ditto approach - optimized for CPU.
"""
import numpy as np
import time
import struct
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from collections import deque

CREDITS = "Powered By DeathLegionTeamLK"

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


@dataclass
class TalkingHeadConfig:
    """Configuration for the Audio-Driven Talking Head"""
    sample_rate: int = 16000
    chunk_size: int = 800  # 50ms at 16kHz
    hop_length: int = 200
    n_mfcc: int = 13
    frame_rate: float = 20.0  # target FPS for output
    
    # Head motion
    head_motion_intensity: float = 0.6
    head_bob_frequency: float = 0.8
    nod_on_emphasis: bool = True
    tilt_on_emphasis: bool = True
    
    # Expression
    expression_smoothing: float = 0.7
    mouth_sensitivity: float = 1.2
    
    # Natural motion patterns
    idle_motion_enabled: bool = True
    idle_motion_amplitude: float = 0.05
    
    # Performance
    use_librosa: bool = True
    cache_features: bool = True


@dataclass
class AudioFeatures:
    """Extracted audio features for a single chunk"""
    mfcc: np.ndarray = None
    pitch: float = 0.0
    energy: float = 0.0
    rms: float = 0.0
    zero_crossing_rate: float = 0.0
    spectral_centroid: float = 0.0
    is_speech: bool = False
    phoneme_category: str = "silence"  # silence, vowel, consonant
    timestamp: float = 0.0
    duration: float = 0.0


@dataclass
class MotionFrame:
    """Single frame of motion output from talking head"""
    head_pose_offset: Dict[str, float] = field(default_factory=lambda: {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0})
    expression_delta: Dict[str, float] = field(default_factory=lambda: {
        'mouth_open': 0.0, 'mouth_smile': 0.0, 'brow_raise': 0.0,
        'eye_open': 1.0, 'jaw_drop': 0.0
    })
    emphasis: float = 0.0
    is_nodding: bool = False
    is_tilting: bool = False
    timestamp: float = 0.0


class AudioFeatureExtractor:
    """Extract audio features (MFCC, pitch, energy) from raw audio chunks."""

    def __init__(self, config: Optional[TalkingHeadConfig] = None):
        self.config = config or TalkingHeadConfig()
        self.use_librosa = self.config.use_librosa and HAS_LIBROSA

    def extract(self, audio_chunk: np.ndarray, sample_rate: Optional[int] = None) -> AudioFeatures:
        """Extract features from an audio chunk."""
        sr = sample_rate or self.config.sample_rate
        af = AudioFeatures(timestamp=time.time(), duration=len(audio_chunk) / sr)

        if len(audio_chunk) == 0:
            return af

        audio_float = audio_chunk.astype(np.float32)
        if np.max(np.abs(audio_float)) > 0:
            audio_float = audio_float / np.max(np.abs(audio_float))

        # Energy (RMS)
        af.rms = float(np.sqrt(np.mean(audio_float ** 2)))
        af.energy = af.rms

        # Pitch via autocorrelation
        af.pitch = self._estimate_pitch(audio_float, sr)

        # Zero crossing rate
        af.zero_crossing_rate = float(np.mean(np.abs(np.diff(np.sign(audio_float)))) / 2.0)

        # Spectral centroid (approximate via energy distribution)
        if len(audio_float) > 1:
            fft = np.abs(np.fft.rfft(audio_float))
            freqs = np.fft.rfftfreq(len(audio_float), 1.0 / sr)
            if np.sum(fft) > 0:
                af.spectral_centroid = float(np.sum(freqs * fft) / np.sum(fft))

        # MFCC using librosa or fallback
        if self.use_librosa and len(audio_float) >= self.config.hop_length * 2:
            try:
                mfcc = librosa.feature.mfcc(y=audio_float, sr=sr, n_mfcc=self.config.n_mfcc,
                                            hop_length=self.config.hop_length)
                if mfcc.shape[1] > 0:
                    af.mfcc = np.mean(mfcc, axis=1)
            except Exception:
                af.mfcc = self._compute_mfcc_fallback(audio_float, sr)
        else:
            af.mfcc = self._compute_mfcc_fallback(audio_float, sr)

        # Speech detection
        af.is_speech = af.rms > 0.02

        # Phoneme category classification
        if af.is_speech:
            if af.zero_crossing_rate > 0.1 and af.spectral_centroid > 2000:
                af.phoneme_category = "consonant"
            else:
                af.phoneme_category = "vowel"
        else:
            af.phoneme_category = "silence"

        return af

    def _estimate_pitch(self, audio: np.ndarray, sr: int) -> float:
        """Estimate fundamental frequency via autocorrelation."""
        if len(audio) < 80:
            return 0.0
        corr = np.correlate(audio, audio, mode='full')
        corr = corr[len(corr) // 2:]
        d = np.diff(corr)
        start = np.where(d > 0)[0]
        if len(start) == 0:
            return 0.0
        peak_idx = np.argmax(corr[start[0]:start[0] + min(500, len(corr) - start[0])]) + start[0]
        if peak_idx > 0:
            pitch = float(sr / peak_idx)
            if 50 < pitch < 500:
                return pitch
        return 0.0

    def _compute_mfcc_fallback(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Manual MFCC computation using FFT filterbank."""
        if len(audio) < 256:
            return np.zeros(self.config.n_mfcc)
        frame_size = min(512, len(audio))
        n_fft = 1
        while n_fft < frame_size:
            n_fft *= 2
        fft = np.abs(np.fft.rfft(audio[:frame_size], n=n_fft))
        fft = fft / (np.sum(fft) + 1e-10)
        # Simple mel-spaced filterbank approximation
        n_filters = 26
        freqs = np.linspace(0, sr / 2, len(fft))
        mel = 2595 * np.log10(1 + freqs / 700)
        mel_bins = np.linspace(mel[0], mel[-1], n_filters + 2)
        mfcc = np.zeros(n_filters)
        for i in range(n_filters):
            start = np.argmin(np.abs(mel - mel_bins[i]))
            center = np.argmin(np.abs(mel - mel_bins[i + 1]))
            end = np.argmin(np.abs(mel - mel_bins[i + 2]))
            if end > center > start:
                mfcc[i] = np.sum(fft[start:end] * np.abs(np.arange(start, end) - center) / (center - start + 1))
        mfcc = np.log(mfcc + 1e-10)
        if len(mfcc) >= self.config.n_mfcc:
            return mfcc[:self.config.n_mfcc]
        return np.pad(mfcc, (0, self.config.n_mfcc - len(mfcc)))


class TalkingHeadEngine:
    """
    Predicts head pose and expressions from audio features.
    Maps audio rhythm/intonation to natural head motion and expression changes.
    """

    def __init__(self, config: Optional[TalkingHeadConfig] = None):
        self.config = config or TalkingHeadConfig()
        self.extractor = AudioFeatureExtractor(config)
        
        # State
        self._phase = 0.0
        self._motion_history = deque(maxlen=30)
        self._expression_history = deque(maxlen=15)
        self._last_features: Optional[AudioFeatures] = None
        
        # Smoothing buffers
        self._smooth_pitch_offset: float = 0.0
        self._smooth_energy_offset: float = 0.0
        self._nod_phase: float = 0.0
        self._tilt_phase: float = 0.0
        self._idle_phase: float = 0.0
        
        # Emphasis detection
        self._energy_peak: float = 0.0
        self._energy_decay: float = 0.0
        self._pitch_peak: float = 0.0
        
        self._frame_count: int = 0

    def process_audio(self, audio_chunk: np.ndarray, sample_rate: Optional[int] = None) -> MotionFrame:
        """Process audio chunk and produce motion frame."""
        features = self.extractor.extract(audio_chunk, sample_rate)
        self._last_features = features
        return self._features_to_motion(features)

    def process_features(self, features: AudioFeatures) -> MotionFrame:
        """Convert pre-extracted audio features to motion."""
        self._last_features = features
        return self._features_to_motion(features)

    def _features_to_motion(self, features: AudioFeatures) -> MotionFrame:
        """Map audio features to head motion and expressions."""
        self._frame_count += 1
        dt = 1.0 / self.config.frame_rate
        self._phase += dt

        mf = MotionFrame(timestamp=features.timestamp)
        energy = features.energy
        pitch = features.pitch
        zcr = features.zero_crossing_rate

        # Emphasis detection
        if energy > self._energy_peak:
            self._energy_peak = energy
            self._energy_decay = 1.0
        else:
            self._energy_decay *= 0.92
            energy *= self._energy_decay

        is_emphasized = energy > 0.15 and self._energy_decay > 0.7

        # Head pitch from energy (nodding)
        self._nod_phase += self.config.head_bob_frequency * dt * 2 * np.pi
        nod_amount = energy * self.config.head_motion_intensity * 15.0
        if is_emphasized and self.config.nod_on_emphasis:
            nod_amount *= 2.5

        mf.head_pose_offset['pitch'] = float(np.sin(self._nod_phase) * nod_amount)
        mf.is_nodding = nod_amount > 2.0

        # Head yaw from pitch changes (turning on intonation)
        if pitch > 100:
            pitch_norm = np.clip((pitch - 100) / 300.0, 0.0, 1.0)
            self._tilt_phase += 0.5 * dt * 2 * np.pi * (1.0 + pitch_norm)
            tilt = np.sin(self._tilt_phase) * pitch_norm * 5.0 * self.config.head_motion_intensity
            if is_emphasized and self.config.tilt_on_emphasis:
                tilt *= 1.5
            mf.head_pose_offset['yaw'] = float(tilt)
            mf.is_tilting = abs(tilt) > 3.0

        # Head roll from energy + zcr
        roll_phase = self._phase * 0.7
        mf.head_pose_offset['roll'] = float(np.sin(roll_phase) * energy * 3.0 * self.config.head_motion_intensity)

        # Idle micro-motion
        if self.config.idle_motion_enabled:
            self._idle_phase += dt * 1.3
            idle_nod = np.sin(self._idle_phase * 1.0) * self.config.idle_motion_amplitude * 3.0
            idle_yaw = np.sin(self._idle_phase * 0.7 + 1.0) * self.config.idle_motion_amplitude * 2.0
            idle_roll = np.sin(self._idle_phase * 0.5 + 2.0) * self.config.idle_motion_amplitude * 1.5
            mf.head_pose_offset['pitch'] += idle_nod
            mf.head_pose_offset['yaw'] += idle_yaw
            mf.head_pose_offset['roll'] += idle_roll

        # Smooth head pose
        alpha = self.config.expression_smoothing
        for k in ['pitch', 'yaw', 'roll']:
            old = getattr(self, f'_smooth_{k}_offset', 0.0)
            val = mf.head_pose_offset[k]
            smoothed = old * alpha + val * (1.0 - alpha)
            mf.head_pose_offset[k] = float(smoothed)
            setattr(self, f'_smooth_{k}_offset', smoothed)

        # Expression deltas
        mouth_open = energy * self.config.mouth_sensitivity * 2.0
        if features.phoneme_category == "vowel":
            mouth_open *= 1.8
        elif features.phoneme_category == "consonant":
            mouth_open *= 0.6

        mf.expression_delta['mouth_open'] = float(np.clip(mouth_open, 0.0, 1.0))
        mf.expression_delta['jaw_drop'] = mf.expression_delta['mouth_open'] * 0.8

        # Smile from spectral centroid (brighter sound = more smile)
        if features.spectral_centroid > 0:
            smile = np.clip((features.spectral_centroid - 1000) / 3000.0, 0.0, 0.6)
            mf.expression_delta['mouth_smile'] = float(smile)

        # Brow raise from pitch (question intonation)
        if pitch > 200:
            brow = np.clip((pitch - 200) / 200.0, 0.0, 0.8)
            mf.expression_delta['brow_raise'] = float(brow)

        # Eye openness (slightly narrowed during speech)
        speech_factor = 1.0 if features.is_speech else 1.0
        mf.expression_delta['eye_open'] = 1.0 - (energy * 0.15 * (1.0 - speech_factor))
        mf.expression_delta['eye_open'] = float(np.clip(mf.expression_delta['eye_open'], 0.7, 1.0))

        # Emphasis flag
        mf.emphasis = float(np.clip(energy * 5.0, 0.0, 1.0))

        # Store in history
        self._motion_history.append(mf)
        self._expression_history.append(mf.expression_delta.copy())

        return mf

    def generate_idle_frames(self, num_frames: int) -> List[MotionFrame]:
        """Generate idle motion frames (no audio input)."""
        frames = []
        silence = np.zeros(self.config.chunk_size)
        for _ in range(num_frames):
            features = self.extractor.extract(silence)
            mf = self._features_to_motion(features)
            mf.emphasis = 0.0
            frames.append(mf)
        return frames

    def audio_to_motion_sequence(self, audio: np.ndarray, sample_rate: int) -> List[MotionFrame]:
        """
        Convert full audio to motion frame sequence.
        
        Args:
            audio: Full audio waveform
            sample_rate: Sample rate of audio
            
        Returns:
            List of MotionFrames at config.frame_rate
        """
        frames = []
        chunk_size = self.config.chunk_size
        hop = chunk_size // 2  # 50% overlap

        for start in range(0, len(audio) - chunk_size + 1, hop):
            chunk = audio[start:start + chunk_size]
            mf = self.process_audio(chunk, sample_rate)
            frames.append(mf)

        return frames

    def get_motion_for_speech(self, energy: float, pitch: float, is_speech: bool = True) -> Dict[str, float]:
        """Quick motion prediction from speech parameters."""
        af = AudioFeatures(energy=energy, pitch=pitch, is_speech=is_speech)
        af.zero_crossing_rate = 0.05 if is_speech else 0.0
        af.spectral_centroid = 1500.0 if is_speech else 500.0
        af.phoneme_category = "vowel" if pitch > 100 else ("consonant" if is_speech else "silence")
        af.timestamp = time.time()
        mf = self.process_features(af)
        return {
            'head_pose': mf.head_pose_offset,
            'expressions': mf.expression_delta,
            'emphasis': mf.emphasis,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            'config': {
                'head_motion_intensity': self.config.head_motion_intensity,
                'expression_smoothing': self.config.expression_smoothing,
                'mouth_sensitivity': self.config.mouth_sensitivity,
                'idle_motion': self.config.idle_motion_enabled,
            },
            'frame_count': self._frame_count,
            'has_librosa': HAS_LIBROSA,
            'credits': CREDITS,
        }


_talking_head_instance: Optional[TalkingHeadEngine] = None


def get_talking_head() -> TalkingHeadEngine:
    global _talking_head_instance
    if _talking_head_instance is None:
        _talking_head_instance = TalkingHeadEngine()
    return _talking_head_instance