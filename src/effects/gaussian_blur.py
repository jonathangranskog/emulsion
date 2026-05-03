from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from src.effects.base import ImageEffect


class GaussianBlur(ImageEffect):
    """Gaussian blur effect that blurs the entire image"""

    def __init__(
        self,
        radius: float = 0.01,
        strength: float = 1.0,
    ):
        super().__init__()
        self.radius = (
            radius  # Normalized radius (0.0 - 0.05) as fraction of max dimension
        )
        self.strength = strength

    def get_params(self) -> List[Dict[str, Any]]:
        return [
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
                "name": "strength",
                "label": "Strength",
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
            },
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur effect using Torch processing

        Args:
            x: Input tensor with shape (C, H, W) in range [0, 1]

        Returns:
            Output tensor with blur applied
        """
        # Compute pixel radius from normalized value
        C, H, W = x.shape
        max_dim = max(H, W)
        radius_pixels = max(1, int(torch.ceil(torch.tensor(self.radius * max_dim))))

        # Create 1D Gaussian blur kernels for separable convolution
        kernel_size = 2 * radius_pixels + 1
        sigma = radius_pixels / 3.0  # 3-sigma rule

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
        image_batched = x.unsqueeze(0)  # (1, 3, H, W)

        # Horizontal blur with groups=3 for per-channel convolution
        # Use replicate padding to match GLSL texture clamping behavior
        h_padded = F.pad(
            image_batched, (radius_pixels, radius_pixels, 0, 0), mode="replicate"
        )
        h_blurred = F.conv2d(h_padded, h_kernel, padding=0, groups=3)

        # Vertical blur with replicate padding
        v_padded = F.pad(
            h_blurred, (0, 0, radius_pixels, radius_pixels), mode="replicate"
        )
        blurred = F.conv2d(v_padded, v_kernel, padding=0, groups=3)

        blurred = blurred.squeeze(0)  # (3, H, W)

        # Blend between original and blurred based on strength
        result = x * (1.0 - self.strength) + blurred * self.strength

        return result

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        """Return two GLSL shaders for separable blur: horizontal then vertical"""

        # Pass 1: Horizontal Gaussian blur
        horizontal_shader = """
        vec4 apply_gaussianblur(vec4 color, float radius_normalized, float strength) {
            vec2 texSize = textureSize(main_texture, 0);
            vec2 texelSize = 1.0 / texSize;

            // Compute pixel radius from normalized value
            float max_dim = max(texSize.x, texSize.y);
            int radius = max(1, int(ceil(radius_normalized * max_dim)));

            float sigma = float(radius) / 3.0;  // 3-sigma rule
            float two_sigma_sq = 2.0 * sigma * sigma;

            vec3 result = vec3(0.0);
            float totalWeight = 0.0;

            // Horizontal Gaussian blur
            for (int x = -radius; x <= radius; x++) {
                vec2 offset = vec2(float(x) * texelSize.x, 0.0);
                vec4 sample = texture(main_texture, TexCoord + offset);

                // Gaussian weight
                float weight = exp(-float(x * x) / two_sigma_sq);

                result += sample.rgb * weight;
                totalWeight += weight;
            }

            return vec4(result / totalWeight, 1.0);
        }
        """.strip()

        # Pass 2: Vertical Gaussian blur and blend with original
        vertical_shader = """
        vec4 apply_gaussianblur(vec4 color, float radius_normalized, float strength) {
            vec2 texSize = textureSize(main_texture, 0);
            vec2 texelSize = 1.0 / texSize;

            // Compute pixel radius from normalized value
            float max_dim = max(texSize.x, texSize.y);
            int radius = max(1, int(ceil(radius_normalized * max_dim)));

            float sigma = float(radius) / 3.0;  // 3-sigma rule
            float two_sigma_sq = 2.0 * sigma * sigma;

            vec3 result = vec3(0.0);
            float totalWeight = 0.0;

            // Vertical Gaussian blur
            for (int y = -radius; y <= radius; y++) {
                vec2 offset = vec2(0.0, float(y) * texelSize.y);
                vec4 sample = texture(main_texture, TexCoord + offset);

                // Gaussian weight
                float weight = exp(-float(y * y) / two_sigma_sq);

                result += sample.rgb * weight;
                totalWeight += weight;
            }

            vec3 blurred = result / totalWeight;

            // Blend with original texture based on strength
            vec4 original = texture(original_texture, TexCoord);
            vec3 final_color = mix(original.rgb, blurred, strength);

            return vec4(final_color, 1.0);
        }
        """.strip()

        uniforms = {
            "u_gaussian_blur_strength": self.strength,
            "u_gaussian_blur_radius": self.radius,
        }

        return [horizontal_shader, vertical_shader], uniforms

    def requires_original_texture(self) -> bool:
        """Blur needs access to original texture for strength blending"""
        return True
