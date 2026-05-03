"""Black and white conversion effect with customizable filter color."""

from typing import Any, Dict, List

import torch

from src.effects.base import ImageEffect


class BlackAndWhite(ImageEffect):
    """Convert image to black and white using a customizable filter color.

    The filter color is used as weights in a dot product with the input RGB values,
    allowing for custom weighted grayscale conversions (e.g., red filter, green filter, etc.).
    The output maintains 3 channels by replicating the grayscale value.
    """

    def __init__(self, filter_color: torch.Tensor | None = None):
        """Initialize black and white effect.

        Args:
            filter_color: RGB weights for the dot product. Default is standard luminance weights.
        """
        super().__init__()
        self.filter_color = (
            filter_color
            if filter_color is not None
            else torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)
        )

    def get_params(self) -> List[Dict[str, Any]]:
        """Get parameter definitions for UI."""
        return [
            {
                "name": "filter_color",
                "label": "Filter Color",
                "type": "vec3",
                "default": torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32),
                "min": torch.zeros(3, dtype=torch.float32),
                "max": torch.ones(3, dtype=torch.float32),
                "step": 0.01,
            }
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply black and white conversion using PyTorch.

        Args:
            x: Input tensor of shape (3, H, W) with RGB channels

        Returns:
            Grayscale tensor of shape (3, H, W) with the same value in all channels
        """
        # x shape: (3, H, W)
        # Ensure filter_color is on the same device and dtype as input
        filter_color = self.filter_color.to(x)[:, None, None]

        # Compute grayscale value using dot product with filter color
        # Broadcasting: (3, 1, 1) * (3, H, W) -> sum over channel dimension
        grayscale = (filter_color * x).sum(dim=0)

        # Replicate grayscale value to all 3 channels to maintain RGB output
        return torch.stack([grayscale, grayscale, grayscale], dim=0)

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        """Get GLSL shader code and uniforms for GPU rendering.

        Returns:
            Tuple of (shader_code_list, uniforms_dict)
        """
        glsl_code = """
        vec4 apply_blackandwhite(vec4 color, vec3 filter_color) {
            // Compute grayscale value using dot product with filter color
            float grayscale = dot(color.rgb, filter_color);
            // Replicate to all 3 channels
            return vec4(vec3(grayscale), color.a);
        }
        """
        uniforms = {"u_blackandwhite_filter_color": self.filter_color}
        return [glsl_code], uniforms
