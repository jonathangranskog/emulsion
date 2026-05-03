import torch
import OpenGL.GL as gl

from src.effects.base import ImageEffect
from src.interface.shader import get_passthrough_shader_program
from typing import List, Dict, Any


class Crop(ImageEffect):
    """
    Crop effect that reduces image dimensions.
    Crop is applied before other effects for better performance.
    """

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        width: int = None,
        height: int = None,
        source_width: int = None,
        source_height: int = None,
    ):
        super().__init__()
        # Validate and clamp all parameters to safe ranges
        self.x = max(0, x) if isinstance(x, int) else 0
        self.y = max(0, y) if isinstance(y, int) else 0
        self.width = (
            max(0, width) if width is not None and isinstance(width, int) else 0
        )
        self.height = (
            max(0, height) if height is not None and isinstance(height, int) else 0
        )
        self.source_width = (
            max(0, source_width)
            if source_width is not None and isinstance(source_width, int)
            else 0
        )
        self.source_height = (
            max(0, source_height)
            if source_height is not None and isinstance(source_height, int)
            else 0
        )

    def is_active(self) -> bool:
        """Check if crop is actually cropping anything"""
        if self.width == 0 or self.height == 0:
            return False
        if self.source_width == 0 or self.source_height == 0:
            return False
        # Check if crop rect equals source dimensions
        if (
            self.x == 0
            and self.y == 0
            and self.width == self.source_width
            and self.height == self.source_height
        ):
            return False
        return True

    def get_params(self) -> List[Dict[str, Any]]:
        # Use safe defaults (will be initialized on first apply if needed)
        source_w = max(1, self.source_width) if self.source_width > 0 else 10000
        source_h = max(1, self.source_height) if self.source_height > 0 else 10000
        default_w = self.source_width if self.source_width > 0 else 10000
        default_h = self.source_height if self.source_height > 0 else 10000

        return [
            {
                "name": "x",
                "label": "Crop X",
                "type": "int",
                "default": 0,
                "min": 0,
                "max": source_w,
                "step": 1,
                "requires_reconstruction": True,
            },
            {
                "name": "y",
                "label": "Crop Y",
                "type": "int",
                "default": 0,
                "min": 0,
                "max": source_h,
                "step": 1,
                "requires_reconstruction": True,
            },
            {
                "name": "width",
                "label": "Crop Width",
                "type": "int",
                "default": default_w,
                "min": 1,
                "max": source_w,
                "step": 1,
                "requires_reconstruction": True,
            },
            {
                "name": "height",
                "label": "Crop Height",
                "type": "int",
                "default": default_h,
                "min": 1,
                "max": source_h,
                "step": 1,
                "requires_reconstruction": True,
            },
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply crop in torch mode by slicing tensor"""
        # Auto-initialize source dimensions from first image ONLY if both are zero
        _, h, w = x.shape
        if self.source_width == 0 and self.source_height == 0:
            self.source_width = w
            self.source_height = h
            # Also initialize crop dimensions to full image
            self.width = w
            self.height = h

        # IMPORTANT: Don't override width/height if they were explicitly set!
        # Only initialize if BOTH are zero (meaning they were never set)
        if self.width == 0 and self.height == 0:
            self.width = self.source_width
            self.height = self.source_height

        if not self.is_active():
            return x

        # Clamp crop rect to image bounds
        x1 = max(0, min(self.x, w))
        y1 = max(0, min(self.y, h))
        x2 = max(0, min(self.x + self.width, w))
        y2 = max(0, min(self.y + self.height, h))

        # Ensure we have valid dimensions
        if x2 <= x1 or y2 <= y1:
            return x

        return x[:, y1:y2, x1:x2]

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        # Shader that samples only the cropped portion of the input
        # NOTE: Within FBO pipeline, TexCoord is already in correct orientation (no Y-flip needed)
        # NOTE: Arguments after 'color' are sorted ALPHABETICALLY by shader processor:
        # u_crop_height, u_crop_width, u_crop_x, u_crop_y, u_source_height, u_source_width
        shader_code = """
vec4 apply_crop(vec4 color, int u_crop_height, int u_crop_width, int u_crop_x, int u_crop_y, int u_source_height, int u_source_width) {
    // TexCoord is in [0, 1] for the output (cropped) FBO
    // Calculate the position in the cropped output (in pixels)
    vec2 output_pixel = TexCoord * vec2(float(u_crop_width), float(u_crop_height));

    // Map to input texture coordinates (in pixel space)
    vec2 input_pixel = output_pixel + vec2(float(u_crop_x), float(u_crop_y));

    // Normalize to [0, 1] for texture sampling
    vec2 input_uv = input_pixel / vec2(float(u_source_width), float(u_source_height));

    // Sample from the input texture at the mapped coordinates
    return texture(main_texture, input_uv);
}
"""
        # Calculate ACTUAL clamped output dimensions (matching get_effect_dimension_deltas)
        source_w = max(1, self.source_width)
        source_h = max(1, self.source_height)

        # Clamp to actual output size (can't extend beyond source bounds)
        actual_crop_w = min(max(1, self.width), source_w - self.x)
        actual_crop_h = min(max(1, self.height), source_h - self.y)

        return [shader_code], {
            "u_crop_x": self.x,
            "u_crop_y": self.y,
            "u_crop_width": actual_crop_w,
            "u_crop_height": actual_crop_h,
            "u_source_width": source_w,
            "u_source_height": source_h,
        }

    def get_effect_dimension_deltas(self) -> tuple[float, float]:
        """Return normalized deltas (fraction of max dimension)"""
        if not self.is_active():
            return (0.0, 0.0)

        # Ensure source dimensions are valid
        source_w = max(1, self.source_width)
        source_h = max(1, self.source_height)

        # Clamp crop parameters to valid ranges
        x = max(0, min(self.x, source_w - 1))
        y = max(0, min(self.y, source_h - 1))
        width = max(1, min(self.width, source_w))
        height = max(1, min(self.height, source_h))

        # Calculate actual output dimensions after clamping (matching torch mode logic)
        # The output can't extend beyond source bounds
        actual_output_width = min(width, source_w - x)
        actual_output_height = min(height, source_h - y)

        # Clamp to at least 1 pixel
        actual_output_width = max(1, actual_output_width)
        actual_output_height = max(1, actual_output_height)

        # Calculate pixel deltas
        delta_w_pixels = actual_output_width - source_w
        delta_h_pixels = actual_output_height - source_h

        # Convert to normalized deltas (fraction of max dimension)
        # Shader processor expects normalized values, not pixel values
        max_dim = max(source_w, source_h)
        if max_dim > 0:
            delta_w_norm = delta_w_pixels / max_dim
            delta_h_norm = delta_h_pixels / max_dim
        else:
            delta_w_norm = 0.0
            delta_h_norm = 0.0

        return (delta_w_norm, delta_h_norm)

    def serialize_to_cache(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "source_width": self.source_width,
            "source_height": self.source_height,
        }

    @classmethod
    def deserialize_from_cache(cls, state: Dict[str, Any]) -> "Crop":
        # Validate and clamp all cached values to safe ranges
        def safe_int(value, default=0):
            """Safely convert to int and ensure non-negative"""
            try:
                result = int(value)
                return max(0, result)
            except (ValueError, TypeError, OverflowError):
                return default

        return cls(
            x=safe_int(state.get("x", 0)),
            y=safe_int(state.get("y", 0)),
            width=safe_int(state.get("width", 0)),
            height=safe_int(state.get("height", 0)),
            source_width=safe_int(state.get("source_width", 0)),
            source_height=safe_int(state.get("source_height", 0)),
        )
