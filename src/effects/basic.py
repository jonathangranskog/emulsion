from typing import Any, Dict, List

import torch

from src.effects.base import ImageEffect


class Exposure(ImageEffect):
    def __init__(self, stops: float = 0.0):
        super().__init__()
        self.stops = stops

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "stops",
                "label": "Stops",
                "type": "float",
                "default": 0.0,
                "min": -5.0,
                "max": 5.0,
                "step": 0.1,
            }
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x * (2.0**self.stops)

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        glsl_code = """
        vec4 apply_exposure(vec4 color, float exposure_stops) {
            vec3 rgb = color.rgb * pow(2.0, exposure_stops);
            return vec4(rgb, color.a);
        }
        """

        uniforms = {"u_exposure_stops": self.stops}

        return [glsl_code], uniforms


class Gamma(ImageEffect):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "gamma",
                "label": "Gamma",
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 5.0,
                "step": 0.01,
            }
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # Only clamp negative values to avoid domain errors in power operation
        return torch.clamp(x, min=0.0) ** (1.0 / max(self.gamma, 1e-6))

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        glsl_code = """
        vec4 apply_gamma(vec4 color, float gamma) {
            // Only clamp negative values to avoid domain errors in power operation
            vec3 rgb = pow(max(color.rgb, vec3(0.0)), vec3(1.0 / max(gamma, 0.000001)));
            return vec4(rgb, color.a);
        }
        """

        uniforms = {"u_gamma": self.gamma}

        return [glsl_code], uniforms


class Contrast(ImageEffect):
    def __init__(self, amount: float = 0.0):
        super().__init__()
        self.amount = amount  # -1..1

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "amount",
                "label": "Contrast",
                "type": "float",
                "default": 0.0,
                "min": -1.0,
                "max": 1.0,
                "step": 0.01,
            }
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # simple pivot at 0.5 in display space
        return (x - 0.5) * (1.0 + self.amount) + 0.5

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        glsl_code = """
        vec4 apply_contrast(vec4 color, float contrast) {
            vec3 rgb = (color.rgb - 0.5) * (1.0 + contrast) + 0.5;
            return vec4(rgb, color.a);
        }
        """

        uniforms = {"u_contrast_amount": self.amount}

        return [glsl_code], uniforms


class Saturation(ImageEffect):
    def __init__(self, amount: float = 0.0):
        super().__init__()
        self.amount = amount  # -1..1 (0=no change, 1=double sat)

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "amount",
                "label": "Saturation",
                "type": "float",
                "default": 0.0,
                "min": -1.0,
                "max": 1.0,
                "step": 0.01,
            }
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[0], x[1], x[2]
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        fac = 1.0 + self.amount
        rs = luma + (r - luma) * fac
        gs = luma + (g - luma) * fac
        bs = luma + (b - luma) * fac
        return torch.stack([rs, gs, bs], dim=0)

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        glsl_code = """
        vec4 apply_saturation(vec4 color, float saturation) {
            // Luminance calculation using standard coefficients
            float luma = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));

            // Apply saturation factor
            float fac = 1.0 + saturation;
            vec3 saturated = luma + (color.rgb - luma) * fac;

            return vec4(saturated, color.a);
        }
        """

        uniforms = {"u_saturation_amount": self.amount}

        return [glsl_code], uniforms


class Vibrance(ImageEffect):
    def __init__(self, amount: float = 0.0):
        super().__init__()
        self.amount = amount  # -1..1 (0=no change, 1=double vibrance)

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "amount",
                "label": "Vibrance",
                "type": "float",
                "default": 0.0,
                "min": -1.0,
                "max": 1.0,
                "step": 0.01,
            }
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[0], x[1], x[2]
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b

        # Calculate current saturation (max - min of RGB)
        max_rgb = torch.maximum(torch.maximum(r, g), b)
        min_rgb = torch.minimum(torch.minimum(r, g), b)
        current_sat = max_rgb - min_rgb

        # Vibrance applies more effect to less saturated colors
        # Use (1 - saturation) as a mask to protect already saturated areas
        sat_mask = torch.clamp(1.0 - current_sat, 0.0, 1.0)

        # Calculate vibrance factor (stronger on less saturated pixels)
        vibrance_factor = 1.0 + (self.amount * sat_mask)

        # Apply vibrance
        rv = luma + (r - luma) * vibrance_factor
        gv = luma + (g - luma) * vibrance_factor
        bv = luma + (b - luma) * vibrance_factor

        return torch.stack([rv, gv, bv], dim=0)

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        glsl_code = """
        vec4 apply_vibrance(vec4 color, float vibrance) {
            // Luminance calculation using standard coefficients
            float luma = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));

            // Calculate current saturation
            float maxRGB = max(max(color.r, color.g), color.b);
            float minRGB = min(min(color.r, color.g), color.b);
            float currentSat = maxRGB - minRGB;

            // Vibrance mask - stronger effect on less saturated pixels
            float satMask = clamp(1.0 - currentSat, 0.0, 1.0);

            // Calculate vibrance factor
            float vibranceFactor = 1.0 + (vibrance * satMask);

            // Apply vibrance
            vec3 vibrant = luma + (color.rgb - luma) * vibranceFactor;

            return vec4(vibrant, color.a);
        }
        """

        uniforms = {"u_vibrance_amount": self.amount}

        return [glsl_code], uniforms


class ColorShift(ImageEffect):
    def __init__(self, shift: torch.Tensor | None = None, scale: float = 0.0):
        super().__init__()
        self.shift = shift if shift is not None else torch.zeros(3, dtype=torch.float32)
        self.scale = scale

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "shift",
                "label": "Color",
                "type": "vec3",
                "default": torch.zeros(3, dtype=torch.float32),
                "min": torch.zeros(3, dtype=torch.float32),
                "max": torch.ones(3, dtype=torch.float32),
                "step": 0.01,
            },
            {
                "name": "scale",
                "label": "Amount",
                "type": "float",
                "default": 0.0,
                "min": -1.0,
                "max": 1.0,
                "step": 0.01,
            },
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # Apply scaled color shift
        shift = (self.shift.to(x) * self.scale)[:, None, None]
        return x + shift

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        glsl_code = """
        vec4 apply_colorshift(vec4 color, float scale, vec3 shift) {
            return vec4(color.rgb + shift * scale, color.a);
        }
        """

        uniforms = {"u_scale": self.scale, "u_shift": self.shift}
        return [glsl_code], uniforms


class Highlights(ImageEffect):
    """Adjust highlights using luminance-based masking with optional color shift"""

    def __init__(
        self, amount: float = 0.0, color: torch.Tensor | None = None, range: float = 4.0
    ):
        super().__init__()
        self.amount = amount  # -1 to 1: darken/brighten highlights
        self.range = range  # 0.5 to 5.0: controls luma mask selectivity
        self.color = (
            color
            if color is not None
            else torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        )

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "amount",
                "label": "Highlights",
                "type": "float",
                "default": 0.0,
                "min": -1.0,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "range",
                "label": "Highlight Range",
                "type": "float",
                "default": 4.0,
                "min": 0.5,
                "max": 5.0,
                "step": 0.1,
            },
            {
                "name": "color",
                "label": "Highlight Color",
                "type": "vec3",
                "default": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
                "min": torch.zeros(3, dtype=torch.float32),
                "max": torch.ones(3, dtype=torch.float32),
                "step": 0.01,
            },
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[0], x[1], x[2]
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        highlight_mask = torch.clamp(torch.pow(luma, self.range), 0.0, 1.0)
        highlight_mult = torch.pow(
            torch.tensor(2.0, dtype=x.dtype, device=x.device), self.amount
        )
        # Apply brightness adjustment and color tint
        color_tint = self.color.to(x)[:, None, None]
        highlight_adjustment = x * highlight_mult * color_tint
        result = x * (1.0 - highlight_mask) + highlight_adjustment * highlight_mask
        return result

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        glsl_code = """
        vec4 apply_highlights(vec4 color, float amount, float range, vec3 tint) {
            float luma = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
            float mask = clamp(pow(luma, range), 0.0, 1.0);
            float mult = pow(2.0, amount);
            vec3 adjustment = color.rgb * mult * tint;
            vec3 result = mix(color.rgb, adjustment, mask);
            return vec4(result, color.a);
        }
        """

        uniforms = {
            "u_highlights_amount": self.amount,
            "u_highlights_range": self.range,
            "u_highlights_tint": self.color,
        }

        return [glsl_code], uniforms


class Shadows(ImageEffect):
    """Adjust shadows using luminance-based masking with optional color shift"""

    def __init__(
        self, amount: float = 0.0, color: torch.Tensor | None = None, range: float = 4.0
    ):
        super().__init__()
        self.amount = amount  # -1 to 1: darken/brighten shadows
        self.range = range  # 0.5 to 5.0: controls luma mask selectivity
        self.color = (
            color
            if color is not None
            else torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        )

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "amount",
                "label": "Shadows",
                "type": "float",
                "default": 0.0,
                "min": -1.0,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "range",
                "label": "Shadow Range",
                "type": "float",
                "default": 4.0,
                "min": 0.5,
                "max": 5.0,
                "step": 0.1,
            },
            {
                "name": "color",
                "label": "Shadow Color",
                "type": "vec3",
                "default": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
                "min": torch.zeros(3, dtype=torch.float32),
                "max": torch.ones(3, dtype=torch.float32),
                "step": 0.01,
            },
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[0], x[1], x[2]
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        shadow_mask = torch.clamp(
            torch.pow(torch.clamp(1.0 - luma, min=0.0), self.range), 0.0, 1.0
        )
        shadow_mult = torch.pow(
            torch.tensor(2.0, dtype=x.dtype, device=x.device), self.amount
        )
        # Apply brightness adjustment and color tint
        color_tint = self.color.to(x)[:, None, None]
        shadow_adjustment = x * shadow_mult * color_tint
        result = x * (1.0 - shadow_mask) + shadow_adjustment * shadow_mask

        return result

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        glsl_code = """
        vec4 apply_shadows(vec4 color, float amount, float range, vec3 tint) {
            float luma = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
            float mask = clamp(pow(max(1.0 - luma, 0.0), range), 0.0, 1.0);
            float mult = pow(2.0, amount);
            vec3 adjustment = color.rgb * mult * tint;
            vec3 result = mix(color.rgb, adjustment, mask);
            return vec4(result, color.a);
        }
        """

        uniforms = {
            "u_shadows_amount": self.amount,
            "u_shadows_range": self.range,
            "u_shadows_tint": self.color,
        }

        return [glsl_code], uniforms
