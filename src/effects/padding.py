import torch

import torch.nn.functional as F

from src.effects.base import ImageEffect
from src.interface.shader import get_passthrough_shader_program
from typing import List, Dict, Any

import OpenGL.GL as gl


class Padding(ImageEffect):
    def __init__(self, pixels: float = 0.0, color: torch.Tensor = None):
        super().__init__()
        self.pixels = pixels  # Normalized (0.0-0.05) as fraction of max dimension
        self.color = color if color is not None else torch.ones(3, dtype=torch.float32)

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "pixels",
                "label": "Padding",
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 0.05,
                "step": 0.001,
                "requires_reconstruction": True,
            },
            {
                "name": "color",
                "label": "Padding Color",
                "type": "vec3",
                "default": torch.ones(3, dtype=torch.float32),
                "min": torch.zeros(3, dtype=torch.float32),
                "max": torch.ones(3, dtype=torch.float32),
                "step": 0.01,
            },
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # Torch mode: pad and fill borders with color
        if self.pixels == 0.0:
            return x

        # Compute pixel padding from normalized value
        C, H, W = x.shape
        max_dim = max(H, W)
        padding_pixels = max(1, int(torch.ceil(torch.tensor(self.pixels * max_dim))))

        # Pad with zeros first
        padded = F.pad(
            x,
            (padding_pixels, padding_pixels, padding_pixels, padding_pixels),
            mode="constant",
            value=0.0,
        )

        # Create mask for border regions and fill with color
        color = self.color.to(x)[:, None, None]
        mask = torch.ones_like(padded)
        mask[:, padding_pixels:-padding_pixels, padding_pixels:-padding_pixels] = 0.0

        result = padded * (1.0 - mask) + color * mask
        return result

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        # No shader code - padding handled specially by processor
        return [""], {"u_padding_pixels": self.pixels, "u_padding_color": self.color}

    def get_effect_dimension_deltas(self) -> tuple[float, float]:
        # Return normalized deltas (fraction of max dimension)
        return (2 * self.pixels, 2 * self.pixels)

    def custom_render(self, input_texture_id: int, input_width: int, input_height: int):
        """Render padding by clearing larger FBO to color and rendering input centered"""
        # Compute pixel padding from normalized value
        max_dim = max(input_width, input_height)
        padding_pixels = max(
            1, int(torch.ceil(torch.tensor(self.pixels * max_dim)).item())
        )

        color = self.color
        gl.glClearColor(float(color[0]), float(color[1]), float(color[2]), 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Now render input texture centered with viewport offset
        gl.glViewport(padding_pixels, padding_pixels, input_width, input_height)

        # Render input texture (fills the centered viewport)
        gl.glUseProgram(get_passthrough_shader_program())
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, input_texture_id)
        gl.glUniform1i(
            gl.glGetUniformLocation(get_passthrough_shader_program(), "main_texture"), 0
        )
