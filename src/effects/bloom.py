from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from src.effects.base import ImageEffect


class Bloom(ImageEffect):
    """Bloom/glow effect that thresholds bright pixels, blurs them, and blends back"""

    def __init__(
        self,
        threshold: float = 0.8,
        intensity: float = 0.5,
        color: torch.Tensor | None = None,
        radius: float = 0.01,
        spread: float = 0.333,
    ):
        super().__init__()
        self.threshold = threshold
        self.intensity = intensity
        self.color = (
            color
            if color is not None
            else torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        )
        self.radius = (
            radius  # Normalized radius (0.0 - 0.05) as fraction of max dimension
        )
        self.spread = spread

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "threshold",
                "label": "Threshold",
                "type": "float",
                "default": 0.8,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "intensity",
                "label": "Intensity",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 3.0,
                "step": 0.01,
            },
            {
                "name": "color",
                "label": "Color",
                "type": "vec3",
                "default": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
                "min": torch.zeros(3, dtype=torch.float32),
                "max": torch.ones(3, dtype=torch.float32),
                "step": 0.01,
            },
            {
                "name": "radius",
                "label": "Radius",
                "type": "float",
                "default": 0.01,
                "min": 0.0,
                "max": 0.05,
                "step": 0.001,
            },
            {
                "name": "spread",
                "label": "Spread",
                "type": "float",
                "default": 0.333,
                "min": 0.1,
                "max": 2.0,
                "step": 0.01,
            },
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bloom effect using Torch processing

        Args:
            x: Input tensor with shape (C, H, W) in range [0, 1]

        Returns:
            Output tensor with bloom applied
        """
        # Calculate luminance for each pixel
        r, g, b = x[0], x[1], x[2]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

        # Threshold to isolate bright areas
        # Create a mask for pixels above threshold
        bright_mask = (luminance >= self.threshold).float()

        # Extract bright pixels (preserve color, but only where bright)
        bright_pixels = x * bright_mask.unsqueeze(0)

        # Compute pixel radius from normalized value
        C, H, W = x.shape
        max_dim = max(H, W)
        radius_pixels = max(1, int(torch.ceil(torch.tensor(self.radius * max_dim))))

        # Create 1D Gaussian blur kernels for separable convolution
        kernel_size = 2 * radius_pixels + 1
        sigma = radius_pixels * self.spread  # Spread controls the Gaussian width

        # Generate 1D Gaussian weights
        weights_1d = torch.zeros(kernel_size, device=x.device, dtype=x.dtype)
        for i in range(kernel_size):
            x_pos = i - radius_pixels
            weights_1d[i] = torch.exp(
                torch.tensor(-x_pos * x_pos / (2.0 * sigma * sigma))
            )
        weights_1d = weights_1d / weights_1d.sum()  # Normalize

        # Horizontal kernel: shape (3, 1, 1, kernel_size)
        # 3 output channels, 1 input channel per group, height=1, width=kernel_size
        h_kernel = weights_1d.view(1, 1, 1, kernel_size).repeat(3, 1, 1, 1)

        # Vertical kernel: shape (3, 1, kernel_size, 1)
        v_kernel = weights_1d.view(1, 1, kernel_size, 1).repeat(3, 1, 1, 1)

        # Apply separable blur (horizontal then vertical)
        # Need to add batch dimension for conv2d
        bright_pixels_batched = bright_pixels.unsqueeze(0)  # (1, 3, H, W)

        # Horizontal blur with groups=3 for per-channel convolution
        # Use replicate padding to match GLSL texture clamping behavior
        h_padded = F.pad(
            bright_pixels_batched,
            (radius_pixels, radius_pixels, 0, 0),
            mode="replicate",
        )
        h_blurred = F.conv2d(h_padded, h_kernel, padding=0, groups=3)

        # Vertical blur with replicate padding
        v_padded = F.pad(
            h_blurred, (0, 0, radius_pixels, radius_pixels), mode="replicate"
        )
        blurred = F.conv2d(v_padded, v_kernel, padding=0, groups=3)

        blurred = blurred.squeeze(0)  # (3, H, W)

        # Blend blurred bright areas with original image
        # Apply color tint and intensity
        color = self.color.to(x.device).to(x.dtype)[:, None, None]  # (3, 1, 1)
        result = x + blurred * color * self.intensity

        return result

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        """Return two GLSL shaders for separable blur: horizontal then vertical"""

        # Pass 1: Horizontal Gaussian blur with threshold
        horizontal_shader = """
        vec4 apply_bloom(vec4 color, vec3 bloom_color, float intensity, float radius_normalized, float spread, float threshold) {
            vec2 texSize = textureSize(main_texture, 0);
            vec2 texelSize = 1.0 / texSize;

            // Compute pixel radius from normalized value
            float max_dim = max(texSize.x, texSize.y);
            int radius = max(1, int(ceil(radius_normalized * max_dim)));

            float sigma = float(radius) * spread;  // Spread controls Gaussian width
            float two_sigma_sq = 2.0 * sigma * sigma;

            vec3 result = vec3(0.0);
            float totalWeight = 0.0;

            // Horizontal Gaussian blur with threshold
            for (int x = -radius; x <= radius; x++) {
                vec2 offset = vec2(float(x) * texelSize.x, 0.0);
                vec4 sample = texture(main_texture, TexCoord + offset);

                // Threshold based on luminance
                float luma = dot(sample.rgb, vec3(0.2126, 0.7152, 0.0722));
                float bright = step(threshold, luma);

                // Gaussian weight
                float weight = exp(-float(x * x) / two_sigma_sq);

                result += sample.rgb * bright * weight;
                totalWeight += weight;
            }

            return vec4(result / totalWeight, 1.0);
        }
        """.strip()

        # Pass 2: Vertical Gaussian blur and blend with original
        vertical_shader = """
        vec4 apply_bloom(vec4 color, vec3 bloom_color, float intensity, float radius_normalized, float spread, float threshold) {
            vec2 texSize = textureSize(main_texture, 0);
            vec2 texelSize = 1.0 / texSize;

            // Compute pixel radius from normalized value
            float max_dim = max(texSize.x, texSize.y);
            int radius = max(1, int(ceil(radius_normalized * max_dim)));

            float sigma = float(radius) * spread;  // Spread controls Gaussian width
            float two_sigma_sq = 2.0 * sigma * sigma;

            vec3 result = vec3(0.0);
            float totalWeight = 0.0;

            // Vertical Gaussian blur (input is already thresholded from horizontal pass)
            for (int y = -radius; y <= radius; y++) {
                vec2 offset = vec2(0.0, float(y) * texelSize.y);
                vec4 sample = texture(main_texture, TexCoord + offset);

                // Gaussian weight
                float weight = exp(-float(y * y) / two_sigma_sq);

                result += sample.rgb * weight;
                totalWeight += weight;
            }

            vec3 blurred = result / totalWeight;

            // Blend with original texture from effect's initial input
            vec4 original = texture(original_texture, TexCoord);
            float luma = dot(original.rgb, vec3(0.2126, 0.7152, 0.0722));
            vec3 final_color = original.rgb + blurred * bloom_color * intensity;

            return vec4(final_color, 1.0);
        }
        """.strip()

        uniforms = {
            "u_bloom_color": self.color,
            "u_bloom_intensity": self.intensity,
            "u_bloom_radius": self.radius,
            "u_bloom_threshold": self.threshold,
            "u_bloom_spread": self.spread,
        }

        return [horizontal_shader, vertical_shader], uniforms

    def requires_original_texture(self) -> bool:
        """Bloom needs access to original texture for final blend"""
        return True
