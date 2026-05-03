"""Vignette effect that darkens the corners and edges of the frame."""

from typing import Any, Dict, List
import torch
from src.effects.base import ImageEffect


class Vignette(ImageEffect):
    def __init__(
        self,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        feather: float = 0.5,
        strength: float = 0.5,
    ):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.feather = feather
        self.strength = strength

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "x_scale",
                "label": "X Scale",
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 2.0,
                "step": 0.01,
            },
            {
                "name": "y_scale",
                "label": "Y Scale",
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 2.0,
                "step": 0.01,
            },
            {
                "name": "feather",
                "label": "Feather",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 2.0,
                "step": 0.01,
            },
            {
                "name": "strength",
                "label": "Strength",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
            },
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        C, H, W = x.shape

        # Use minimum dimension for circular vignettes by default
        min_res = min(H, W)
        y_coords = torch.linspace(0.0, H / min_res, H, device=x.device)
        x_coords = torch.linspace(0.0, W / min_res, W, device=x.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        xx_centered = xx - (W / (2.0 * min_res))
        yy_centered = yy - (H / (2.0 * min_res))
        xx_scaled = xx_centered / self.x_scale
        yy_scaled = yy_centered / self.y_scale
        distance = torch.sqrt(xx_scaled**2 + yy_scaled**2)

        # Apply smooth falloff with feathering
        # Using smooth step function for gradual transition
        edge_start = 1.0 - self.feather
        edge_end = 1.0 + self.feather

        # Clamp and apply smoothstep
        t = torch.clamp((distance - edge_start) / (edge_end - edge_start), 0.0, 1.0)
        vignette_mask = t * t * (3.0 - 2.0 * t)  # Smoothstep formula

        # Apply strength and create final mask
        darkening = vignette_mask * self.strength
        result = x * (1.0 - darkening).unsqueeze(0)
        return result

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        shader_code = """
        vec4 apply_vignette(vec4 color, float feather, float strength,
                                   float x_scale, float y_scale) {
            vec2 texSize = vec2(textureSize(main_texture, 0));
            float minRes = min(texSize.x, texSize.y);
            vec2 uv = gl_FragCoord.xy / minRes;
            vec2 centered = uv - texSize / (2.0 * minRes);
            vec2 scaled = vec2(centered.x / x_scale, centered.y / y_scale);
            float dist = length(scaled);
            float edge_start = 1.0 - feather;
            float edge_end = 1.0 + feather;
            float vignette_mask = smoothstep(edge_start, edge_end, dist);
            float darkening = vignette_mask * strength;
            vec3 result = color.rgb * (1.0 - darkening);
            return vec4(result, color.a);
        }
        """

        uniforms = {
            "u_x_scale": self.x_scale,
            "u_y_scale": self.y_scale,
            "u_feather": self.feather,
            "u_strength": self.strength,
        }

        return [shader_code], uniforms
