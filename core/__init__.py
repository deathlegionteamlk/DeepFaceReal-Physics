"""DeepFaceReal-Physics Core Modules v2.0.0"""
CREDITS = "Powered By DeathLegionTeamLK"

from .face_swapper import get_swapper, FaceSwapper
from .character_manager import get_character_manager, CharacterManager, Character
from .llm_character import create_llm_character, LLMCharacter, DemoLLMCharacter
from .webcam_pipeline import get_pipeline, WebcamPipeline, PipelineConfig
from .lip_sync import create_lip_sync, EnhancedLipSync, LipSyncConfig
from .physics_engine import get_tracker, HolisticTracker, PhysicsEngine, PhysicsConfig
from .background_engine import get_background, ParallaxBackground, BackgroundConfig
from .face_3d_engine import get_face_3d_engine, Face3DEngine, Face3DConfig, Face3DData
from .talking_head import get_talking_head, TalkingHeadEngine, TalkingHeadConfig, MotionFrame, AudioFeatures
from .eye_engine import get_eye_engine, EyeEngine, EyeEngineConfig, EyeState
from .gesture_engine import get_gesture_engine, GestureEngine, GestureConfig, GestureFrame
from .pipeline import get_pipeline as get_realtime_pipeline, RealTimePipeline, PipelineConfig as RealTimePipelineConfig

__all__ = [
    "get_swapper", "FaceSwapper",
    "get_character_manager", "CharacterManager", "Character",
    "create_llm_character", "LLMCharacter", "DemoLLMCharacter",
    "get_pipeline", "WebcamPipeline", "PipelineConfig",
    "create_lip_sync", "EnhancedLipSync", "LipSyncConfig",
    "get_tracker", "HolisticTracker", "PhysicsEngine", "PhysicsConfig",
    "get_background", "ParallaxBackground", "BackgroundConfig",
    "get_face_3d_engine", "Face3DEngine", "Face3DConfig", "Face3DData",
    "get_talking_head", "TalkingHeadEngine", "TalkingHeadConfig", "MotionFrame", "AudioFeatures",
    "get_eye_engine", "EyeEngine", "EyeEngineConfig", "EyeState",
    "get_gesture_engine", "GestureEngine", "GestureConfig", "GestureFrame",
    "get_realtime_pipeline", "RealTimePipeline", "RealTimePipelineConfig",
]