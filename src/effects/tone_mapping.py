"""
Tone mapping operators for HDR to SDR conversion.

These operators compress high dynamic range images into displayable [0,1] range
while preserving details and maintaining a pleasing appearance.
"""

from typing import Any, Dict, List

import torch

from src.effects.base import ImageEffect


class ReinhardToneMapping(ImageEffect):
    """
    Reinhard tone mapping operator.

    Simple and effective tone mapping that compresses HDR values smoothly.
    Formula: L_out = L_in / (1 + L_in)

    This asymptotically approaches 1.0 as input increases, preserving details
    in both shadows and highlights.
    """

    def __init__(self, white_point: float = 2.0):
        super().__init__()
        self.white_point = white_point

    @classmethod
    def get_params(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "white_point",
                "label": "White Point",
                "type": "float",
                "default": 2.0,
                "min": 1.0,
                "max": 10.0,
                "step": 0.1,
            }
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # Reinhard tone mapping with adjustable white point
        # L_display = L * (1 + L/L_white^2) / (1 + L)
        white_sq = self.white_point * self.white_point
        return x * (1.0 + x / white_sq) / (1.0 + x)

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        glsl_code = """
        vec4 apply_reinhardtonemapping(vec4 color, float white_point) {
            vec3 rgb = color.rgb;
            float white_sq = white_point * white_point;
            rgb = rgb * (1.0 + rgb / white_sq) / (1.0 + rgb);
            return vec4(rgb, color.a);
        }
        """

        uniforms = {"u_white_point": self.white_point}

        return [glsl_code], uniforms


class ACESFilmicToneMapping(ImageEffect):
    """
    ACES Filmic tone mapping operator.

    Industry-standard tone mapping used in film and game production.
    Provides a cinematic look with smooth roll-off in highlights.

    Based on Stephen Hill's ACES approximation:
    https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    """

    def __init__(self, exposure: float = 1.0):
        super().__init__()
        self.exposure = exposure

    @classmethod
    def get_params(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "exposure",
                "label": "Exposure",
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 5.0,
                "step": 0.1,
            }
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # Apply exposure adjustment
        x = x * self.exposure

        # ACES filmic tone mapping curve
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14

        result = (x * (a * x + b)) / (x * (c * x + d) + e)
        return result

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        glsl_code = """
        vec4 apply_acesfilmictonemapping(vec4 color, float exposure) {
            vec3 rgb = color.rgb * exposure;

            // ACES filmic tone mapping curve
            float a = 2.51;
            float b = 0.03;
            float c = 2.43;
            float d = 0.59;
            float e = 0.14;

            rgb = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e);
            return vec4(rgb, color.a);
        }
        """

        uniforms = {"u_exposure": self.exposure}

        return [glsl_code], uniforms
