"""Temperature and tint color adjustment effect.

Uses Planckian locus color science for physically-based temperature
adjustments when RAW metadata is available, with a simple fallback
for non-RAW images.
"""

import math
from typing import Any, Dict, List

import torch

from src.core.metadata import ImageMetadata
from src.effects.base import ImageEffect

# Default reference temperature (D65 daylight illuminant)
DEFAULT_REFERENCE_KELVIN = 6500.0

# Maximum mired shift at slider extremes.  150 mireds from a 6500 K
# reference gives roughly 3 300 K (warm tungsten) at +1 and a very high
# Kelvin (extreme cool blue) at -1, which is a useful creative range.
MIRED_RANGE = 150.0

# Hard limits for target Kelvin to avoid numerical issues.
KELVIN_MIN = 1000.0
KELVIN_MAX = 40000.0

# Tint strength: maximum green-channel multiplicative shift at slider=+/-1.
TINT_STRENGTH = 0.3


def kelvin_to_rgb(kelvin: float) -> tuple[float, float, float]:
    """Convert a color temperature in Kelvin to normalised linear-sRGB values.

    Based on a piecewise fit to the CIE 1931 2-degree standard observer
    applied to the Planckian (blackbody) locus, originally by Tanner
    Helland and later refined.  The returned RGB represents the *colour*
    of a blackbody radiator at the given temperature.

    The output is normalised so that the brightest channel is 1.0.

    Valid range: ~1000 K - ~40 000 K.
    """
    temp = kelvin / 100.0

    # --- Red ---
    if temp <= 66.0:
        red = 255.0
    else:
        red = 329.698727446 * ((temp - 60.0) ** -0.1332047592)
        red = max(0.0, min(255.0, red))

    # --- Green ---
    if temp <= 66.0:
        green = 99.4708025861 * math.log(temp) - 161.1195681661
        green = max(0.0, min(255.0, green))
    else:
        green = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)
        green = max(0.0, min(255.0, green))

    # --- Blue ---
    if temp >= 66.0:
        blue = 255.0
    elif temp <= 19.0:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(temp - 10.0) - 305.0447927307
        blue = max(0.0, min(255.0, blue))

    # Normalise so the brightest channel equals 1.0.
    mx = max(red, green, blue, 1e-10)
    return red / mx, green / mx, blue / mx


def _compute_rgb_correction(
    temperature: float,
    tint: float,
    reference_kelvin: float,
) -> tuple[float, float, float]:
    """Compute per-channel RGB multipliers for a temperature/tint adjustment.

    Parameters
    ----------
    temperature : float
        Slider value in [-1, 1].  Positive = warmer, negative = cooler.
    tint : float
        Slider value in [-1, 1].  Positive = green, negative = magenta.
    reference_kelvin : float
        The "neutral" colour temperature in Kelvin (e.g. as-shot CCT).

    Returns
    -------
    (r_mul, g_mul, b_mul) : multiplicative correction factors.
    """
    # --- Temperature via mired mapping ---
    ref_mired = 1e6 / reference_kelvin
    target_mired = ref_mired + temperature * MIRED_RANGE
    # Clamp to valid Kelvin range
    target_mired = max(1e6 / KELVIN_MAX, min(1e6 / KELVIN_MIN, target_mired))
    target_kelvin = 1e6 / target_mired

    ref_r, ref_g, ref_b = kelvin_to_rgb(reference_kelvin)
    tgt_r, tgt_g, tgt_b = kelvin_to_rgb(target_kelvin)

    # Ratio gives the multiplicative correction.
    eps = 1e-10
    temp_r = tgt_r / max(ref_r, eps)
    temp_g = tgt_g / max(ref_g, eps)
    temp_b = tgt_b / max(ref_b, eps)

    # --- Tint (green-magenta axis, perpendicular to Planckian locus) ---
    # Positive tint → boost green, attenuate R & B (green shift)
    # Negative tint → attenuate green, boost R & B (magenta shift)
    tint_g = 1.0 + tint * TINT_STRENGTH
    # Split the inverse equally across R and B to roughly preserve luminance.
    tint_rb = 1.0 - tint * TINT_STRENGTH * 0.5

    # Combine
    r_mul = temp_r * tint_rb
    g_mul = temp_g * tint_g
    b_mul = temp_b * tint_rb

    return r_mul, g_mul, b_mul


class TemperatureTint(ImageEffect):
    """Adjust image temperature (cool/warm) and tint (magenta/green).

    When RAW metadata is available the effect uses the as-shot colour
    temperature as a reference and applies physically-based corrections
    derived from the Planckian (blackbody) locus.  The slider maps
    through *mired* space (reciprocal mega-kelvins) which is
    perceptually uniform.

    When no RAW metadata is present the reference defaults to 6 500 K
    (D65 daylight), which still gives a well-behaved Planckian-based
    correction.

    Values are not clamped to preserve HDR range.
    """

    def __init__(self, temperature: float = 0.0, tint: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.tint = tint

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "temperature",
                "label": "Temperature",
                "type": "float",
                "default": 0.0,
                "min": -1.0,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "tint",
                "label": "Tint",
                "type": "float",
                "default": 0.0,
                "min": -1.0,
                "max": 1.0,
                "step": 0.01,
            },
        ]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temperature and tint adjustment.

        Temperature shifts the blue-orange axis via the Planckian locus:
        - Positive: warmer (shift toward lower Kelvin / more orange)
        - Negative: cooler (shift toward higher Kelvin / more blue)

        Tint shifts the green-magenta axis (perpendicular to the locus):
        - Positive: add green
        - Negative: add magenta

        Args:
            x: Input image tensor (C, H, W) in linear RGB space

        Returns:
            Adjusted image tensor (not clamped to preserve HDR)
        """
        as_shot_kelvin = ImageMetadata.get(
            "raw_as_shot_kelvin", DEFAULT_REFERENCE_KELVIN
        )
        r_mul, g_mul, b_mul = _compute_rgb_correction(
            self.temperature, self.tint, as_shot_kelvin
        )

        correction = torch.tensor(
            [r_mul, g_mul, b_mul], dtype=x.dtype, device=x.device
        )[:, None, None]

        return x * correction

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        """Get GLSL shader code and uniforms.

        The RGB correction is computed on the CPU (via the Planckian locus
        utilities) and passed as a vec3 uniform so the shader is a single
        per-pixel multiply.
        """
        as_shot_kelvin = ImageMetadata.get(
            "raw_as_shot_kelvin", DEFAULT_REFERENCE_KELVIN
        )
        r_mul, g_mul, b_mul = _compute_rgb_correction(
            self.temperature, self.tint, as_shot_kelvin
        )

        glsl_code = """
        vec4 apply_temperaturetint(vec4 color, vec3 rgb_correction) {
            return vec4(color.rgb * rgb_correction, color.a);
        }
        """

        uniforms = {
            "u_rgb_correction": [r_mul, g_mul, b_mul],
        }
        return [glsl_code], uniforms
