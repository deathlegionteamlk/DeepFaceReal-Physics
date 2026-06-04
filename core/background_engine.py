import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

CREDITS = 'Powered By DeathLegionTeamLK'


@dataclass
class BackgroundConfig:
    mode: str = 'static'
    layer_count: int = 3
    shift_speed: List[float] = field(default_factory=lambda: [0.15, 0.45, 0.80])
    blur_amount: float = 5.0
    blur_enabled: bool = True
    near_color: Tuple[int, int, int] = (30, 40, 60)
    mid_color: Tuple[int, int, int] = (20, 30, 50)
    far_color: Tuple[int, int, int] = (10, 15, 25)
    accent_color: Tuple[int, int, int] = (60, 90, 140)
    width: int = 640
    height: int = 480


class ParallaxBackground:
    def __init__(self, config: Optional[BackgroundConfig] = None):
        self.config = config or BackgroundConfig()
        self._base_layers = []
        self._time = 0.0
        self._generate_base_layers()

    def _generate_base_layers(self):
        self._base_layers = []
        w, h = self.config.width, self.config.height
        colors = [self.config.far_color, self.config.mid_color, self.config.near_color]
        for layer_idx in range(self.config.layer_count):
            base = np.zeros((h, w, 3), dtype=np.uint8)
            base[:] = colors[layer_idx]
            if layer_idx == 0:
                cv2.rectangle(base, (0, 0), (w, h), self._darken(colors[layer_idx], 30), -1)
                grad = np.tile(np.linspace(0.7, 1.0, w), (h, 1))
                for c in range(3):
                    base[:, :, c] = (base[:, :, c] * grad).astype(np.uint8)
            elif layer_idx == 1:
                center_x, center_y = w // 2, h // 2
                for r in range(50, min(w, h) // 3, 30):
                    alpha = max(0, 1.0 - r / (min(w, h) // 3))
                    cv2.circle(base, (center_x, center_y), r, self._lighten(colors[layer_idx], 20), 2)
            else:
                for i in range(3, 8):
                    x = (w * i) // 10
                    y = (h * i) // 10
                    cv2.circle(base, (x, y), 8 + i * 3, self._lighten(colors[layer_idx], 30), -1)
                    cv2.circle(base, (x, y), 8 + i * 3 + 2, self._lighten(colors[layer_idx], 15), 1)
            self._base_layers.append(base)

    def _darken(self, color: Tuple[int, int, int], amount: int) -> Tuple[int, int, int]:
        return tuple(max(0, c - amount) for c in color)

    def _lighten(self, color: Tuple[int, int, int], amount: int) -> Tuple[int, int, int]:
        return tuple(min(255, c + amount) for c in color)

    def render(self, head_x: float = 0.0, head_y: float = 0.0, overlay: Optional[np.ndarray] = None) -> np.ndarray:
        if self.config.mode == 'static':
            return self._base_layers[0].copy() if overlay is None else overlay

        self._time += 0.02
        w, h = self.config.width, self.config.height
        result = np.zeros((h, w, 3), dtype=np.uint8)
        shift_speeds = self.config.shift_speed

        for layer_idx in range(self.config.layer_count):
            speed = shift_speeds[min(layer_idx, len(shift_speeds) - 1)]
            offset_x = int(head_x * speed * 2.0)
            offset_y = int(head_y * speed * 2.0)
            layer = self._base_layers[layer_idx].copy()
            if self.config.mode == 'parallax':
                M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                layer = cv2.warpAffine(layer, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            elif self.config.mode == 'blur':
                if self.config.blur_enabled and layer_idx < self.config.layer_count - 1:
                    blur_size = int(max(3, self.config.blur_amount * (1.0 - layer_idx / self.config.layer_count) * 2 + 1))
                    if blur_size % 2 == 0:
                        blur_size += 1
                    layer = cv2.GaussianBlur(layer, (blur_size, blur_size), 0)
            result = cv2.addWeighted(result, 0.5, layer, 0.5, 0)

        if overlay is not None:
            mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask = cv2.GaussianBlur(mask, (5, 5), 2)
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = (result * (1 - mask_3ch) + overlay * mask_3ch).astype(np.uint8)

        return result

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        if any(k in kwargs for k in ['mode', 'layer_count']):
            self._generate_base_layers()

    def get_config(self) -> dict:
        return {
            'mode': self.config.mode,
            'layer_count': self.config.layer_count,
            'shift_speed': self.config.shift_speed,
            'blur_amount': self.config.blur_amount,
            'blur_enabled': self.config.blur_enabled,
            'width': self.config.width,
            'height': self.config.height,
        }

    def resize(self, width: int, height: int):
        self.config.width = width
        self.config.height = height
        self._generate_base_layers()


_background_instance = None


def get_background() -> ParallaxBackground:
    global _background_instance
    if _background_instance is None:
        _background_instance = ParallaxBackground()
    return _background_instance