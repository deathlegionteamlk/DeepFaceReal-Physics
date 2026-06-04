
import os
import json
import cv2
import numpy as np
from typing import Optional, List, Dict, Any, Any as AnyType
from datetime import datetime

CREDITS = 'Powered By DeathLegionTeamLK'

CHARACTERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'profiles')

class Character:

    def __init__(
        self,
        name: str = "Unknown",
        photo_path: str = "",
        system_prompt: str = "",
        voice_enabled: bool = False,
        voice_speed: float = 1.0,
        voice_pitch: float = 1.0,
        swap_enabled: bool = True,
        enhance_quality: str = "medium",
        blend_enabled: bool = True
    ):
        self.name = name
        self.photo_path = photo_path
        self.system_prompt = system_prompt or self._default_prompt(name)
        self.voice_enabled = voice_enabled
        self.voice_speed = voice_speed
        self.voice_pitch = voice_pitch
        self.swap_enabled = swap_enabled
        self.enhance_quality = enhance_quality
        self.blend_enabled = blend_enabled
        self.face_embedding: Optional[np.ndarray] = None
        self.source_face: Any = None
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

    def _default_prompt(self, name: str) -> str:
        return f"""You are {name}, a character in a conversation.
Respond naturally and conversationally as {name} would.
Keep responses concise (1-3 sentences) for real-time conversation.
Stay in character at all times. Be engaging and responsive to the user."""

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'name': self.name,
            'photo_path': self.photo_path,
            'system_prompt': self.system_prompt,
            'voice_enabled': self.voice_enabled,
            'voice_speed': self.voice_speed,
            'voice_pitch': self.voice_pitch,
            'swap_enabled': self.swap_enabled,
            'enhance_quality': self.enhance_quality,
            'blend_enabled': self.blend_enabled,
            'created_at': self.created_at,
            'updated_at': datetime.now().isoformat(),
            'has_embedding': self.face_embedding is not None
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Character':
        char = cls(
            name=data.get('name', 'Unknown'),
            photo_path=data.get('photo_path', ''),
            system_prompt=data.get('system_prompt', ''),
            voice_enabled=data.get('voice_enabled', False),
            voice_speed=data.get('voice_speed', 1.0),
            voice_pitch=data.get('voice_pitch', 1.0),
            swap_enabled=data.get('swap_enabled', True),
            enhance_quality=data.get('enhance_quality', 'medium'),
            blend_enabled=data.get('blend_enabled', True)
        )
        char.created_at = data.get('created_at', char.created_at)
        char.updated_at = data.get('updated_at', char.updated_at)
        return char

class CharacterManager:

    def __init__(self, profiles_dir: str = None):
        self.profiles_dir = profiles_dir or CHARACTERS_DIR
        os.makedirs(self.profiles_dir, exist_ok=True)
        self.characters: Dict[str, Character] = {}
        self.active_character_name: Optional[str] = None
        self._load_all()

    def _get_profile_path(self, name: str) -> str:
        safe_name = name.replace(' ', '_').replace('/', '_')
        return os.path.join(self.profiles_dir, f"{safe_name}.json")

    def create_character(
        self,
        name: str,
        photo_path: str = "",
        system_prompt: str = "",
        voice_enabled: bool = False
    ) -> Character:
        char = Character(
            name=name,
            photo_path=photo_path,
            system_prompt=system_prompt,
            voice_enabled=voice_enabled
        )
        self.characters[name] = char
        self.save_character(name)
        return char

    def save_character(self, name: str) -> bool:
        if name not in self.characters:
            return False
        char = self.characters[name]
        path = self._get_profile_path(name)
        with open(path, 'w') as f:
            json.dump(char.to_dict(), f, indent=2)
        return True

    def load_character(self, name: str) -> Optional[Character]:
        path = self._get_profile_path(name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            char = Character.from_dict(data)
            self.characters[name] = char
            return char
        except Exception as e:
            print(f"[CharacterManager] Error loading {name}: {e}")
            return None

    def delete_character(self, name: str) -> bool:
        if name in self.characters:
            del self.characters[name]
        path = self._get_profile_path(name)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def _load_all(self):
        if not os.path.exists(self.profiles_dir):
            return
        for fname in os.listdir(self.profiles_dir):
            if fname.endswith('.json'):
                name = fname.replace('.json', '').replace('_', ' ')
                self.load_character(name)

    def list_characters(self) -> List[str]:
        return list(self.characters.keys())

    def get_character(self, name: str) -> Optional[Character]:
        if name in self.characters:
            return self.characters[name]
        return self.load_character(name)

    def set_active_character(self, name: str) -> bool:
        char = self.get_character(name)
        if char:
            self.active_character_name = name
            return True
        return False

    def get_active_character(self) -> Optional[Character]:
        if self.active_character_name:
            return self.get_character(self.active_character_name)
        return None

    def set_character_face(self, name: str, photo_path: str, face_embedding: np.ndarray, source_face: Any = None):
        char = self.get_character(name)
        if char:
            char.photo_path = photo_path
            char.face_embedding = face_embedding
            char.source_face = source_face
            self.save_character(name)

    def get_character_summaries(self) -> List[Dict[str, Any]]:
        summaries = []
        for name, char in self.characters.items():
            summaries.append({
                'name': char.name,
                'has_photo': bool(char.photo_path and os.path.exists(char.photo_path)),
                'has_embedding': char.face_embedding is not None,
                'voice_enabled': char.voice_enabled,
                'swap_enabled': char.swap_enabled,
                'enhance_quality': char.enhance_quality,
            })
        return summaries

_manager_instance: Optional[CharacterManager] = None

def get_character_manager() -> CharacterManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = CharacterManager()
    return _manager_instance