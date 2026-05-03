"""
Effect registry for searchable effect creation.

This module provides a registry of all available effects that can be
instantiated through the search interface.
"""

import torch

from src.effects.basic import (
    ColorShift,
    Contrast,
    Exposure,
    Gamma,
    Highlights,
    Saturation,
    Shadows,
    Vibrance,
)
from src.effects.black_and_white import BlackAndWhite
from src.effects.bloom import Bloom
from src.effects.gaussian_blur import GaussianBlur
from src.effects.noise_blur import NoiseBlur
from src.effects.lut import LUT, SearchLUT
from src.effects.padding import Padding
from src.effects.crop import Crop
from src.effects.vignette import Vignette
from src.effects.tone_mapping import (
    ACESFilmicToneMapping,
    ReinhardToneMapping,
)
from src.effects.gaussian_grain import GaussianGrain
from src.effects.texture_overlay import TextureOverlay
from src.effects.temperature_tint import TemperatureTint

# Only list nicknames for effects that not the the same as class name
EFFECT_NICKNAMES = {
    "GaussianGrain": ["grain", "noise"],
    "ColorShift": ["color shift"],
    "GaussianBlur": ["blur"],
    "NoiseBlur": ["variable blur", "turbulent blur", "noise blur"],
    "BlackAndWhite": ["bw", "grayscale", "monochrome", "black and white"],
    "ReinhardToneMapping": ["reinhard", "tone mapping"],
    "ACESFilmicToneMapping": ["filmic", "aces", "tone mapping"],
    "TextureOverlay": ["texture", "overlay", "texture overlay"],
    "TemperatureTint": ["temperature", "tint", "white balance", "color temp"],
    "SearchLUT": ["search lut", "search"],
}


def get_effect_registry():
    """
    Returns a dictionary mapping effect names (lowercase) to factory functions.

    Returns:
        dict: Mapping of effect names to lambda functions that create effect instances
              with default parameters.
    """

    # We set some default values so that the effects have an impact when added to the stack.
    return {
        "exposure": lambda: Exposure(stops=0.15),
        "gamma": lambda: Gamma(gamma=2.2),
        "contrast": lambda: Contrast(amount=0.1),
        "saturation": lambda: Saturation(amount=0.1),
        "vibrance": lambda: Vibrance(amount=0.1),
        "blackandwhite": lambda: BlackAndWhite(
            filter_color=torch.tensor([0.2126, 0.7152, 0.0722])
        ),
        "colorshift": lambda: ColorShift(shift=torch.ones(3), scale=0.1),
        "highlights": lambda: Highlights(amount=0.2),
        "shadows": lambda: Shadows(amount=0.2),
        "bloom": lambda: Bloom(
            threshold=0.6, intensity=0.2, color=torch.ones(3), radius=0.02
        ),
        "glow": lambda: Bloom(
            threshold=0.6, intensity=0.2, color=torch.ones(3), radius=0.02
        ),
        "blur": lambda: GaussianBlur(radius=0.01, strength=1.0),
        "noiseblur": lambda: NoiseBlur(
            min_radius=0.002,
            max_radius=0.02,
            strength=1.0,
            noise_scale=0.01,
        ),
        "grain": lambda: GaussianGrain(
            density=0.5,
            size_mean=0.25,
            size_std=0.25,
            intensity_mean=0.0,
            intensity_std=0.1,
            color_shift=0.0,
            luma_size_scale=4.0,
            seed=42,
        ),
        "lut": lambda: LUT(lut_path="", strength=1.0),
        "padding": lambda: Padding(pixels=0.02, color=torch.ones(3)),
        "crop": lambda: Crop(x=0, y=0, width=0, height=0),
        "vignette": lambda: Vignette(
            x_scale=1.0, y_scale=1.0, feather=0.5, strength=0.5
        ),
        # Tone mapping operators for HDR to SDR conversion
        "tonemapping": lambda: ReinhardToneMapping(white_point=2.0),
        "aces": lambda: ACESFilmicToneMapping(exposure=1.0),
        "textureoverlay": lambda: TextureOverlay(
            texture_path="",
            blend_mode=0,
            opacity=1.0,
            scale_x=1.0,
            scale_y=1.0,
            flip_x=False,
            flip_y=False,
            use_bw=False,
        ),
        "temperaturetint": lambda: TemperatureTint(temperature=0.0, tint=0.0),
        "searchlut": lambda: SearchLUT(search_query=""),
    }


def full_effect_registry():
    effect_registry = get_effect_registry()
    effect_nicknames = {k.lower(): v for k, v in EFFECT_NICKNAMES.items()}
    new_effect_registry = effect_registry.copy()
    for key in effect_registry:
        for nickname in effect_nicknames.get(key, []):
            new_effect_registry[nickname] = new_effect_registry[key]
    return new_effect_registry
