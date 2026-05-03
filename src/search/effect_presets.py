"""
Effect preset registry for preview generation.

This module provides hard-coded presets for common effects, allowing users
to preview 5 different variations before applying an effect.
"""

import torch
from typing import Dict, List, Any, Callable
from dataclasses import dataclass


@dataclass
class EffectPreset:
    """Represents a single preset configuration for an effect"""

    label: str  # Descriptive name like "Subtle", "Dramatic", etc.
    params: Dict[str, Any]  # Parameter values to pass to effect constructor


class EffectPresetRegistry:
    """
    Registry of hard-coded presets for each effect type.

    Each effect has 5 preset variations that demonstrate the range of the effect.
    The presets are ordered from subtle to strong for most effects.

    Presets are keyed by effect class name (e.g., "Bloom", "GaussianBlur") to
    automatically work with all nicknames and aliases defined in registry.py.
    """

    def __init__(self, effect_registry: Dict[str, Callable] = None):
        """
        Initialize the preset registry.

        Args:
            effect_registry: Optional effect registry to build search term mappings.
                           If provided, will automatically resolve nicknames.
        """
        self.effect_registry = effect_registry
        self._search_term_to_class = {}  # Maps search terms to class names

        if effect_registry:
            self._build_search_term_mapping()

        # Presets keyed by effect class name
        self.presets: Dict[str, List[EffectPreset]] = {
            # BASIC ADJUSTMENTS
            "Exposure": [
                EffectPreset("Subtle Darken", {"stops": -0.25}),
                EffectPreset("Subtle Brighten", {"stops": 0.25}),
                EffectPreset("Moderate Brighten", {"stops": 0.5}),
                EffectPreset("Strong Brighten", {"stops": 0.75}),
                EffectPreset("Maximum Brighten", {"stops": 1.0}),
            ],
            "Gamma": [
                EffectPreset("Darken", {"gamma": 0.5}),
                EffectPreset("Subtle Darken", {"gamma": 0.7}),
                EffectPreset("Slightly Brighten", {"gamma": 1.4}),
                EffectPreset("Brighten", {"gamma": 1.8}),
                EffectPreset("SRGB", {"gamma": 2.2}),
            ],
            "Contrast": [
                EffectPreset("Low Contrast", {"amount": -0.3}),
                EffectPreset("Subtle Reduction", {"amount": -0.15}),
                EffectPreset("Subtle Increase", {"amount": 0.15}),
                EffectPreset("High Contrast", {"amount": 0.4}),
                EffectPreset("Maximum Contrast", {"amount": 0.7}),
            ],
            "Saturation": [
                EffectPreset("Desaturated", {"amount": -0.5}),
                EffectPreset("Slight Reduction", {"amount": -0.25}),
                EffectPreset("Slight Boost", {"amount": 0.25}),
                EffectPreset("Vibrant", {"amount": 0.5}),
                EffectPreset("Ultra Vibrant", {"amount": 1.0}),
            ],
            "Vibrance": [
                EffectPreset("Subtle Reduction", {"amount": -0.3}),
                EffectPreset("Slight Reduction", {"amount": -0.15}),
                EffectPreset("Natural Boost", {"amount": 0.2}),
                EffectPreset("Strong Boost", {"amount": 0.4}),
                EffectPreset("Maximum Boost", {"amount": 0.7}),
            ],
            "Highlights": [
                EffectPreset("Dim", {"amount": -0.2}),
                EffectPreset("Subtle Dim", {"amount": -0.1}),
                EffectPreset("Subtle Lift", {"amount": 0.1}),
                EffectPreset("Lift", {"amount": 0.2}),
                EffectPreset("Strong Lift", {"amount": 0.3}),
            ],
            "Shadows": [
                EffectPreset("Subtle Dim", {"amount": -0.1}),
                EffectPreset("Dim", {"amount": -0.2}),
                EffectPreset("Subtle Lift", {"amount": 0.1}),
                EffectPreset("Lift", {"amount": 0.2}),
                EffectPreset("Strong Lift", {"amount": 0.3}),
            ],
            # BLUR EFFECTS
            "GaussianBlur": [
                EffectPreset("Soft", {"radius": 0.001, "strength": 1.0}),
                EffectPreset("Light", {"radius": 0.002, "strength": 1.0}),
                EffectPreset("Medium", {"radius": 0.005, "strength": 1.0}),
                EffectPreset("Strong", {"radius": 0.0075, "strength": 1.0}),
                EffectPreset("Heavy", {"radius": 0.01, "strength": 1.0}),
            ],
            "NoiseBlur": [
                EffectPreset(
                    "Subtle Turbulence",
                    {
                        "min_radius": 0.001,
                        "max_radius": 0.01,
                        "strength": 1.0,
                        "noise_scale": 0.01,
                    },
                ),
                EffectPreset(
                    "Light Turbulence",
                    {
                        "min_radius": 0.002,
                        "max_radius": 0.02,
                        "strength": 1.0,
                        "noise_scale": 0.015,
                    },
                ),
                EffectPreset(
                    "Medium Turbulence",
                    {
                        "min_radius": 0.003,
                        "max_radius": 0.03,
                        "strength": 1.0,
                        "noise_scale": 0.02,
                    },
                ),
                EffectPreset(
                    "Strong Turbulence",
                    {
                        "min_radius": 0.005,
                        "max_radius": 0.04,
                        "strength": 1.0,
                        "noise_scale": 0.025,
                    },
                ),
                EffectPreset(
                    "Extreme Turbulence",
                    {
                        "min_radius": 0.01,
                        "max_radius": 0.05,
                        "strength": 1.0,
                        "noise_scale": 0.03,
                    },
                ),
            ],
            # ARTISTIC EFFECTS
            "Bloom": [
                EffectPreset(
                    "Subtle Glow",
                    {
                        "threshold": 0.8,
                        "intensity": 0.10,
                        "color": torch.ones(3),
                        "radius": 0.01,
                    },
                ),
                EffectPreset(
                    "Soft Bloom",
                    {
                        "threshold": 0.8,
                        "intensity": 0.2,
                        "color": torch.ones(3),
                        "radius": 0.02,
                    },
                ),
                EffectPreset(
                    "Medium Bloom",
                    {
                        "threshold": 0.7,
                        "intensity": 0.3,
                        "color": torch.ones(3),
                        "radius": 0.02,
                    },
                ),
                EffectPreset(
                    "Strong Bloom",
                    {
                        "threshold": 0.8,
                        "intensity": 0.5,
                        "color": torch.ones(3),
                        "radius": 0.05,
                    },
                ),
                EffectPreset(
                    "Red Glow",
                    {
                        "threshold": 0.8,
                        "intensity": 0.15,
                        "color": torch.tensor([1.0, 0.0, 0.0]),
                        "radius": 0.05,
                    },
                ),
            ],
            "GaussianGrain": [
                EffectPreset(
                    "Fine Grain",
                    {
                        "density": 0.3,
                        "size_mean": 0.15,
                        "size_std": 0.1,
                        "intensity_mean": 0.0,
                        "intensity_std": 0.06,
                        "color_shift": 0.2,
                        "luma_size_scale": 4.0,
                        "seed": 42,
                    },
                ),
                EffectPreset(
                    "Light Grain",
                    {
                        "density": 0.5,
                        "size_mean": 0.25,
                        "size_std": 0.15,
                        "intensity_mean": 0.0,
                        "intensity_std": 0.1,
                        "color_shift": 0.2,
                        "luma_size_scale": 4.0,
                        "seed": 42,
                    },
                ),
                EffectPreset(
                    "Medium Grain",
                    {
                        "density": 0.7,
                        "size_mean": 0.35,
                        "size_std": 0.2,
                        "intensity_mean": 0.0,
                        "intensity_std": 0.15,
                        "color_shift": 0.0,
                        "luma_size_scale": 4.0,
                        "seed": 42,
                    },
                ),
                EffectPreset(
                    "Heavy Grain",
                    {
                        "density": 0.85,
                        "size_mean": 0.5,
                        "size_std": 0.25,
                        "intensity_mean": 0.0,
                        "intensity_std": 0.2,
                        "color_shift": 0.4,
                        "luma_size_scale": 4.0,
                        "seed": 42,
                    },
                ),
                EffectPreset(
                    "Film Stock",
                    {
                        "density": 1.0,
                        "size_mean": 0.6,
                        "size_std": 0.3,
                        "intensity_mean": 0.0,
                        "intensity_std": 0.25,
                        "color_shift": 0.5,
                        "luma_size_scale": 4.0,
                        "seed": 42,
                    },
                ),
            ],
            "Vignette": [
                EffectPreset(
                    "Subtle Edge",
                    {"x_scale": 1.0, "y_scale": 1.0, "feather": 0.7, "strength": 0.2},
                ),
                EffectPreset(
                    "Light Vignette",
                    {"x_scale": 1.0, "y_scale": 1.0, "feather": 0.6, "strength": 0.4},
                ),
                EffectPreset(
                    "Medium Vignette",
                    {"x_scale": 1.0, "y_scale": 1.0, "feather": 0.5, "strength": 0.6},
                ),
                EffectPreset(
                    "Strong Vignette",
                    {"x_scale": 1.0, "y_scale": 1.0, "feather": 0.4, "strength": 0.8},
                ),
                EffectPreset(
                    "Dramatic",
                    {"x_scale": 1.0, "y_scale": 1.0, "feather": 0.3, "strength": 1.0},
                ),
            ],
            "BlackAndWhite": [
                EffectPreset(
                    "Natural",
                    {"filter_color": torch.tensor([0.2126, 0.7152, 0.0722])},
                ),
                EffectPreset(
                    "Red Filter", {"filter_color": torch.tensor([0.5, 0.3, 0.2])}
                ),
                EffectPreset(
                    "Green Filter", {"filter_color": torch.tensor([0.2, 0.6, 0.2])}
                ),
                EffectPreset(
                    "Blue Filter", {"filter_color": torch.tensor([0.15, 0.25, 0.6])}
                ),
                EffectPreset(
                    "High Contrast", {"filter_color": torch.tensor([0.3, 0.59, 0.11])}
                ),
            ],
            "Padding": [
                EffectPreset("Subtle White Padding", {"pixels": 0.005}),
                EffectPreset("Light Padding", {"pixels": 0.01}),
                EffectPreset("Medium Padding", {"pixels": 0.02}),
                EffectPreset(
                    "Subtle Black Padding", {"pixels": 0.005, "color": torch.zeros(3)}
                ),
                EffectPreset(
                    "Light Black Padding", {"pixels": 0.01, "color": torch.zeros(3)}
                ),
            ],
            "ColorShift": [
                EffectPreset(
                    "Warm Tone",
                    {"shift": torch.tensor([1.0, 0.6, 0.3]), "scale": 0.075},
                ),
                EffectPreset(
                    "Cool Tone",
                    {"shift": torch.tensor([0.3, 0.6, 1.0]), "scale": 0.075},
                ),
                EffectPreset(
                    "Magenta Tint",
                    {"shift": torch.tensor([1.0, 0.3, 0.8]), "scale": 0.06},
                ),
                EffectPreset(
                    "Cyan Tint", {"shift": torch.tensor([0.2, 0.8, 1.0]), "scale": 0.06}
                ),
                EffectPreset(
                    "Green Tint",
                    {"shift": torch.tensor([0.4, 1.0, 0.5]), "scale": 0.06},
                ),
            ],
            "TemperatureTint": [
                EffectPreset("Cool", {"temperature": -0.05, "tint": 0.0}),
                EffectPreset("Warm", {"temperature": 0.05, "tint": 0.0}),
                EffectPreset("Very Warm", {"temperature": 0.1, "tint": 0.0}),
                EffectPreset("Green Tint", {"temperature": 0.0, "tint": 0.05}),
                EffectPreset("Magenta Tint", {"temperature": 0.0, "tint": -0.05}),
            ],
            "ReinhardToneMapping": [
                EffectPreset("Subtle", {"white_point": 1.0}),
                EffectPreset("Light", {"white_point": 1.5}),
                EffectPreset("Medium", {"white_point": 2.0}),
                EffectPreset("Strong", {"white_point": 2.5}),
                EffectPreset("Maximum", {"white_point": 3.0}),
            ],
            "ACESFilmicToneMapping": [
                EffectPreset("Subtle", {"exposure": 0.5}),
                EffectPreset("Light", {"exposure": 1.0}),
                EffectPreset("Medium", {"exposure": 1.5}),
                EffectPreset("Strong", {"exposure": 2.0}),
                EffectPreset("Maximum", {"exposure": 2.5}),
            ],
        }

    def _build_search_term_mapping(self):
        """
        Build a mapping from search terms to effect class names using the effect registry.

        This allows us to resolve all nicknames and aliases automatically without
        maintaining a separate mapping.
        """
        if not self.effect_registry:
            return

        # For each search term in the registry, determine which class it creates
        for search_term, factory in self.effect_registry.items():
            try:
                # Instantiate the effect to get its class name
                effect_instance = factory()
                class_name = effect_instance.__class__.__name__
                self._search_term_to_class[search_term.lower()] = class_name
            except Exception:
                # If instantiation fails, skip this search term
                pass

    def _resolve_to_class_name(self, effect_name: str) -> str:
        """
        Resolve a search term to its effect class name.

        Args:
            effect_name: The effect name or nickname

        Returns:
            The effect class name, or the original name if not found
        """
        effect_name = effect_name.lower()
        return self._search_term_to_class.get(effect_name, effect_name)

    def get_presets(self, effect_name: str) -> List[EffectPreset]:
        """
        Get presets for an effect.

        Args:
            effect_name: The effect name (case-insensitive), can be a nickname

        Returns:
            List of EffectPreset objects, or empty list if no presets defined
        """
        # Resolve search term to effect class name if we have the registry
        if self.effect_registry:
            class_name = self._resolve_to_class_name(effect_name)
        else:
            # Fallback: try the effect name as-is
            class_name = effect_name

        return self.presets.get(class_name, [])

    def has_presets(self, effect_name: str) -> bool:
        """
        Check if an effect has presets defined.

        Args:
            effect_name: The effect name (case-insensitive), can be a nickname

        Returns:
            True if presets exist for this effect
        """
        # Resolve search term to effect class name if we have the registry
        if self.effect_registry:
            class_name = self._resolve_to_class_name(effect_name)
        else:
            # Fallback: try the effect name as-is
            class_name = effect_name

        return class_name in self.presets

    def add_presets(self, effect_name: str, presets: List[EffectPreset]) -> None:
        """
        Add or update presets for an effect. Useful for extending the registry.

        Args:
            effect_name: The effect name (case-insensitive)
            presets: List of EffectPreset objects (should typically be 5 presets)
        """
        self.presets[effect_name.lower()] = presets
