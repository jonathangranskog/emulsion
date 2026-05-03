"""Noise-based variable blur effect."""

from typing import Any, Dict, List
import torch
import torch.nn.functional as F
from src.effects.base import ImageEffect


class NoiseBlur(ImageEffect):
    """Applies Gaussian blur with intensity modulated by simplex noise.

    The blur radius varies across the image based on a simplex noise pattern,
    creating organic, turbulent blur effects.
    """

    def __init__(
        self,
        min_radius: float = 0.001,
        max_radius: float = 0.02,
        strength: float = 1.0,
        noise_scale: float = 0.01,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ):
        """Initialize NoiseBlur effect.

        Args:
            min_radius: Minimum blur radius as fraction of max dimension (in areas with negative noise)
            max_radius: Maximum blur radius as fraction of max dimension (in areas with positive noise)
            strength: Overall effect strength (blend with original)
            noise_scale: Frequency of the noise pattern
            offset_x: Horizontal offset for noise pattern as fraction of min dimension
            offset_y: Vertical offset for noise pattern as fraction of min dimension
        """
        super().__init__()
        self.min_radius = min_radius  # Normalized radius (0.0 - 0.05)
        self.max_radius = max_radius  # Normalized radius (0.0 - 0.05)
        self.strength = strength
        self.noise_scale = noise_scale
        self.offset_x = offset_x
        self.offset_y = offset_y

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "min_radius",
                "label": "Min Radius",
                "type": "float",
                "default": 0.001,
                "min": 0.0,
                "max": 0.05,
                "step": 0.001,
            },
            {
                "name": "max_radius",
                "label": "Max Radius",
                "type": "float",
                "default": 0.02,
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
            {
                "name": "noise_scale",
                "label": "Noise Scale",
                "type": "float",
                "default": 0.1,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "offset_x",
                "label": "Offset X",
                "type": "float",
                "default": 0.0,
                "min": -1.0,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "offset_y",
                "label": "Offset Y",
                "type": "float",
                "default": 0.0,
                "min": -1.0,
                "max": 1.0,
                "step": 0.01,
            },
        ]

    def _generate_simplex_noise(
        self,
        height: int,
        width: int,
        scale: float,
        device: torch.device,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        coord_offset_x: torch.Tensor = None,
        coord_offset_y: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate 2D simplex noise using fully vectorized tensor operations.

        Simplified from SimplexNoise effect - single octave only for performance.
        """

        def mod289(x):
            return x - torch.floor(x * (1.0 / 289.0)) * 289.0

        def permute(x):
            return mod289(((x * 34.0) + 1.0) * x)

        # Create coordinate grids
        y = torch.arange(height, device=device, dtype=torch.float32) + 0.5
        x = torch.arange(width, device=device, dtype=torch.float32) + 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Apply constant offsets
        xx = xx + offset_x
        yy = yy + offset_y

        # Apply per-pixel coordinate offsets for domain warping
        if coord_offset_x is not None:
            xx = xx + coord_offset_x
        if coord_offset_y is not None:
            yy = yy + coord_offset_y

        # Simplex constants
        F2 = 0.366025403784439
        G2 = 0.211324865405187

        # Scale coordinates
        v_x = xx * scale
        v_y = yy * scale

        # First corner (skewed)
        s = (v_x + v_y) * F2
        i = torch.floor(v_x + s)
        j = torch.floor(v_y + s)

        t = (i + j) * G2
        X0 = i - t
        Y0 = j - t
        x0 = v_x - X0
        y0 = v_y - Y0

        # Determine which simplex we're in
        i1 = (x0 > y0).float()
        j1 = (x0 <= y0).float()

        # Offsets for middle corner
        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2

        # Offsets for last corner
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2

        # Permutations
        i = mod289(i)
        j = mod289(j)

        # Calculate permutation indices for three corners
        p0 = permute(permute(j) + i)
        p1 = permute(permute(j + j1) + i + i1)
        p2 = permute(permute(j + 1.0) + i + 1.0)

        # Calculate gradients
        C_www = 0.024390243902439

        # Extract gradient components
        px = torch.stack([p0, p1, p2], dim=-1)
        px_scaled = px * C_www
        gx = 2.0 * (px_scaled - torch.floor(px_scaled)) - 1.0
        gh = torch.abs(gx) - 0.5
        ox = torch.floor(gx + 0.5)
        a0 = gx - ox

        # Calculate contributions from three corners
        m0 = 0.5 - x0 * x0 - y0 * y0
        m1 = 0.5 - x1 * x1 - y1 * y1
        m2 = 0.5 - x2 * x2 - y2 * y2
        m = torch.stack([m0, m1, m2], dim=-1)

        # Clamp negative values to zero
        m = torch.clamp(m, min=0.0)

        # Normalize gradients
        m = m * m
        m = m * m

        # Apply gradient normalization factor
        norm = 1.79284291400159 - 0.85373472095314 * (a0 * a0 + gh * gh)
        m = m * norm

        # Compute gradient dot products
        x_coords = torch.stack([x0, x1, x2], dim=-1)
        y_coords = torch.stack([y0, y1, y2], dim=-1)

        g = a0 * x_coords + gh * y_coords

        # Sum contributions
        noise = 130.0 * torch.sum(m * g, dim=-1)

        return noise

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply noise-based variable blur effect using Torch processing.

        Strategy: Generate multiple blur passes at different radii, then blend
        them based on the noise map for a smooth variable blur approximation.

        Args:
            x: Input tensor with shape (C, H, W) in range [0, 1]

        Returns:
            Output tensor with variable blur applied
        """
        C, H, W = x.shape
        max_dim = max(H, W)

        # Compute pixel radii from normalized values
        min_radius_pixels = max(
            1, int(torch.ceil(torch.tensor(self.min_radius * max_dim)))
        )
        max_radius_pixels = max(
            1, int(torch.ceil(torch.tensor(self.max_radius * max_dim)))
        )

        # Generate noise map (convert normalized offset to pixels using min dimension)
        min_dim = min(H, W)
        noise_map = self._generate_simplex_noise(
            H,
            W,
            self.noise_scale * 0.01,
            x.device,
            offset_x=self.offset_x * min_dim,
            offset_y=self.offset_y * min_dim,
        )

        # Normalize noise from [-1, 1] to [0, 1] range
        noise_normalized = (noise_map + 1.0) * 0.5
        noise_normalized = torch.clamp(noise_normalized, 0.0, 1.0)  # (H, W)

        # Generate multiple blur levels
        # We'll use 5 discrete blur levels and interpolate between them
        num_levels = 5
        blur_levels = []

        for level in range(num_levels):
            # Interpolate radius between min and max
            t = level / (num_levels - 1) if num_levels > 1 else 0.0
            radius = int(
                min_radius_pixels + t * (max_radius_pixels - min_radius_pixels)
            )
            radius = max(1, radius)  # Ensure at least radius 1

            # Apply Gaussian blur at this radius
            kernel_size = 2 * radius + 1
            sigma = radius / 3.0

            # Generate 1D Gaussian weights
            weights_1d = torch.zeros(kernel_size, device=x.device, dtype=x.dtype)
            for i in range(kernel_size):
                x_pos = i - radius
                weights_1d[i] = torch.exp(
                    torch.tensor(-x_pos * x_pos / (2.0 * sigma * sigma))
                )
            weights_1d = weights_1d / weights_1d.sum()

            # Create separable kernels
            h_kernel = weights_1d.view(1, 1, 1, kernel_size).repeat(3, 1, 1, 1)
            v_kernel = weights_1d.view(1, 1, kernel_size, 1).repeat(3, 1, 1, 1)

            # Apply separable blur
            image_batched = x.unsqueeze(0)
            h_padded = F.pad(image_batched, (radius, radius, 0, 0), mode="replicate")
            h_blurred = F.conv2d(h_padded, h_kernel, padding=0, groups=3)

            v_padded = F.pad(h_blurred, (0, 0, radius, radius), mode="replicate")
            blurred = F.conv2d(v_padded, v_kernel, padding=0, groups=3)
            blurred = blurred.squeeze(0)

            blur_levels.append(blurred)

        # Blend blur levels based on noise map
        # noise_normalized is in [0, 1], maps to blur level index [0, num_levels-1]
        level_index = noise_normalized * (num_levels - 1)  # (H, W)
        level_floor = torch.floor(level_index).long()
        level_ceil = torch.ceil(level_index).long()
        level_frac = level_index - level_floor.float()

        # Clamp indices
        level_floor = torch.clamp(level_floor, 0, num_levels - 1)
        level_ceil = torch.clamp(level_ceil, 0, num_levels - 1)

        # Initialize result
        result = torch.zeros_like(x)

        # Blend between floor and ceil levels
        for i in range(num_levels):
            # Create masks for pixels using this level as floor or ceil
            floor_mask = (level_floor == i).float()  # (H, W)
            ceil_mask = (level_ceil == i).float()  # (H, W)

            # Weight by interpolation factor
            floor_weight = floor_mask * (1.0 - level_frac)  # (H, W)
            ceil_weight = ceil_mask * level_frac  # (H, W)

            total_weight = floor_weight + ceil_weight  # (H, W)

            # Add contribution (broadcast across channels)
            result += blur_levels[i] * total_weight.unsqueeze(0)  # (3, H, W)

        # Blend between original and blurred based on strength
        result = x * (1.0 - self.strength) + result * self.strength

        return result

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        """Return two GLSL shaders for separable blur: horizontal then vertical.

        Uses separable Gaussian blur where blur radius is determined per-pixel
        by sampling simplex noise.
        """
        # Shared simplex noise functions for both passes
        noise_functions = """
        // 2D Simplex noise function (from SimplexNoise effect)
        vec3 mod289(vec3 x) {
            return x - floor(x * (1.0 / 289.0)) * 289.0;
        }

        vec2 mod289(vec2 x) {
            return x - floor(x * (1.0 / 289.0)) * 289.0;
        }

        vec3 permute(vec3 x) {
            return mod289(((x * 34.0) + 1.0) * x);
        }

        float snoise(vec2 v) {
            const vec4 C = vec4(0.211324865405187,
                                0.366025403784439,
                               -0.577350269189626,
                                0.024390243902439);

            vec2 i  = floor(v + dot(v, C.yy));
            vec2 x0 = v -   i + dot(i, C.xx);

            vec2 i1;
            i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
            vec4 x12 = x0.xyxy + C.xxzz;
            x12.xy -= i1;

            i = mod289(i);
            vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                                   + i.x + vec3(0.0, i1.x, 1.0));

            vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
            m = m * m;
            m = m * m;

            vec3 x = 2.0 * fract(p * C.www) - 1.0;
            vec3 h = abs(x) - 0.5;
            vec3 ox = floor(x + 0.5);
            vec3 a0 = x - ox;

            m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

            vec3 g;
            g.x  = a0.x  * x0.x  + h.x  * x0.y;
            g.yz = a0.yz * x12.xz + h.yz * x12.yw;
            return 130.0 * dot(m, g);
        }
        """

        # Pass 1: Horizontal blur
        horizontal_shader = (
            noise_functions
            + """
        vec4 apply_noiseblur(vec4 color, float max_radius_normalized, float min_radius_normalized, float noise_scale, float offset_x, float offset_y, float strength) {
            vec2 texSize = textureSize(main_texture, 0);
            vec2 texelSize = 1.0 / texSize;
            vec2 pos = gl_FragCoord.xy;

            // Convert normalized offset to pixels using min dimension
            float min_dim = min(texSize.x, texSize.y);
            vec2 noise_pos = pos + vec2(offset_x, offset_y) * min_dim;

            // Compute pixel radii from normalized values
            float max_dim = max(texSize.x, texSize.y);
            int min_radius = max(1, int(ceil(min_radius_normalized * max_dim)));
            int max_radius = max(1, int(ceil(max_radius_normalized * max_dim)));

            // Generate noise to determine blur radius
            float noise = snoise(noise_pos * noise_scale);

            // Map noise from [-1, 1] to [0, 1]
            float noise_normalized = (noise + 1.0) * 0.5;
            noise_normalized = clamp(noise_normalized, 0.0, 1.0);

            // Determine blur radius based on noise
            float radius_float = mix(float(min_radius), float(max_radius), noise_normalized);
            int radius = int(radius_float);
            radius = max(1, radius);

            // Apply horizontal Gaussian blur
            float sigma = float(radius) / 3.0;
            float two_sigma_sq = 2.0 * sigma * sigma;

            vec3 result = vec3(0.0);
            float totalWeight = 0.0;

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
        )

        # Pass 2: Vertical blur and blend with original
        vertical_shader = (
            noise_functions
            + """
        vec4 apply_noiseblur(vec4 color, float max_radius_normalized, float min_radius_normalized, float noise_scale, float offset_x, float offset_y, float strength) {
            vec2 texSize = textureSize(main_texture, 0);
            vec2 texelSize = 1.0 / texSize;
            vec2 pos = gl_FragCoord.xy;

            // Convert normalized offset to pixels using min dimension
            float min_dim = min(texSize.x, texSize.y);
            vec2 noise_pos = pos + vec2(offset_x, offset_y) * min_dim;

            // Compute pixel radii from normalized values
            float max_dim = max(texSize.x, texSize.y);
            int min_radius = max(1, int(ceil(min_radius_normalized * max_dim)));
            int max_radius = max(1, int(ceil(max_radius_normalized * max_dim)));

            // Generate noise to determine blur radius (same as horizontal pass)
            float noise = snoise(noise_pos * noise_scale);

            // Map noise from [-1, 1] to [0, 1]
            float noise_normalized = (noise + 1.0) * 0.5;
            noise_normalized = clamp(noise_normalized, 0.0, 1.0);

            // Determine blur radius based on noise
            float radius_float = mix(float(min_radius), float(max_radius), noise_normalized);
            int radius = int(radius_float);
            radius = max(1, radius);

            // Apply vertical Gaussian blur
            float sigma = float(radius) / 3.0;
            float two_sigma_sq = 2.0 * sigma * sigma;

            vec3 result = vec3(0.0);
            float totalWeight = 0.0;

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
        )

        uniforms = {
            "u_noise_blur_max_radius": self.max_radius,
            "u_noise_blur_min_radius": self.min_radius,
            "u_noise_blur_noise_scale": self.noise_scale * 0.01,
            "u_noise_blur_strength": self.strength,
            "u_noise_blur_offset_x": self.offset_x,
            "u_noise_blur_offset_y": self.offset_y,
        }

        return [horizontal_shader, vertical_shader], uniforms

    def requires_original_texture(self) -> bool:
        """Noise blur needs access to original texture for strength blending."""
        return True
