import hashlib
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from src.effects.base import ImageEffect
from src.pipeline.texture_cache import TextureDiskCache
from src.pipeline.state_cache import EffectStateCache
from src.grain.gaussian_splatting import generate_grain_texture
from src.interface.texture import TextureManager

# Try to import Metal tiled version, fall back to CPU if not available
try:
    from src.grain.gaussian_splatting_metal_tiled import (
        generate_grain_texture_metal_tiled,
    )
    import platform

    METAL_AVAILABLE = platform.system() == "Darwin"
except ImportError:
    METAL_AVAILABLE = False
    print("Metal not available, using CPU implementation for grain generation")


class GaussianGrain(ImageEffect):
    """Gaussian Splatting Grain Effect

    This effect uses Gaussian Splatting to generate a grain texture for a specific image,
    and then blends it with the original image. The individual grains are randomly distributed,
    and randomly sized inversely proportional to the luminance of the image.

    We use `src.grain.gaussian_splatting.generate_grain_texture` to generate the grain texture.
    The grain texture is automatically rebuilt whenever parameters change, using disk caching
    to avoid redundant generation.
    """

    def __init__(
        self,
        density: int = 0.5,
        size_mean: float = 0.25,
        size_std: float = 0.25,
        intensity_mean: float = 0.25,
        intensity_std: float = 0.25,
        color_shift: float = 0.0,
        luma_size_scale: float = 2.0,
        strength: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.density = density
        self.size_mean = size_mean
        self.size_std = size_std
        self.intensity_mean = intensity_mean
        self.intensity_std = intensity_std
        self.color_shift = color_shift
        self.luma_size_scale = luma_size_scale
        self.strength = strength
        self.seed = 42 if seed is None else seed
        self.grain_texture_name = "gaussian_grain_texture"
        self.grain_texture = None
        self.grain_texture_path = None
        self._texture_resolution = None  # Track resolution texture was built for
        self._shader_warning_shown = False
        self._built_with_params = None  # Track param values used for last build

    @property
    def requires_intermediate_state(self) -> bool:
        return True

    @staticmethod
    def _compute_content_hash(image: torch.Tensor) -> str:
        """Compute a lightweight hash of image content for cache keying."""
        step_y = max(1, image.shape[1] // 32)
        step_x = max(1, image.shape[2] // 32)
        small = image[:, ::step_y, ::step_x].cpu().numpy().copy()
        return hashlib.md5(small.tobytes()).hexdigest()[:16]

    def _build_cache_params(self, width, height, content_hash=None):
        """Build the cache params dict used for texture disk cache keying."""
        params = {
            "width": width,
            "height": height,
            "density": self.density,
            "size_mean": self.size_mean,
            "size_std": self.size_std,
            "intensity_mean": self.intensity_mean,
            "intensity_std": self.intensity_std,
            "color_shift": self.color_shift,
            "luma_size_scale": self.luma_size_scale,
            "seed": self.seed,
        }
        if content_hash is not None:
            params["content_hash"] = content_hash
        return params

    def _current_grain_params(self) -> tuple:
        """Return a tuple of current grain generation parameter values."""
        return (
            self.density,
            self.size_mean,
            self.size_std,
            self.intensity_mean,
            self.intensity_std,
            self.color_shift,
            self.luma_size_scale,
            self.seed,
        )

    def _invalidate_if_params_changed(self):
        """Invalidate grain texture if slider parameters have changed since last build."""
        if self._built_with_params is None:
            return  # No texture built yet, nothing to invalidate

        if self._current_grain_params() != self._built_with_params:
            print("Parameters changed - invalidating grain texture")
            self.grain_texture = None
            self.grain_texture_path = None
            self._texture_resolution = None
            self._built_with_params = None

    def get_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "strength",
                "label": "Strength",
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 5.0,
                "step": 0.01,
            },
            {
                "name": "density",
                "label": "Density",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,  # 1 grain per pixel
                "step": 0.001,
            },
            {
                "name": "size_mean",
                "label": "Size Mean",
                "type": "float",
                "default": 0.25,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "size_std",
                "label": "Size Standard Deviation",
                "type": "float",
                "default": 0.25,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "intensity_mean",
                "label": "Intensity Mean",
                "type": "float",
                "default": 0.0,
                "min": -1.0,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "intensity_std",
                "label": "Intensity Standard Deviation",
                "type": "float",
                "default": 0.25,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "color_shift",
                "label": "Color Shift",
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
            },
            {
                "name": "luma_size_scale",
                "label": "Luma Size Scale",
                "type": "float",
                "default": 2.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.1,
            },
            {
                "name": "seed",
                "label": "Seed",
                "type": "int",
                "default": 42,
                "min": 1,
                "max": 999999,
                "step": 1,
            },
        ]

    def build_grain_texture(self, current_image: torch.Tensor):
        """Build the grain texture and cache it to disk"""
        print(f"build_grain_texture called with image shape: {current_image.shape}")
        rgb_image = current_image.cpu().permute(1, 2, 0).numpy()
        luminance_map = (
            0.2126 * rgb_image[:, :, 0]
            + 0.7152 * rgb_image[:, :, 1]
            + 0.0722 * rgb_image[:, :, 2]
        )
        # Use current_image dimensions for grain texture to ensure luminance map
        # coordinates align with grain positions. The grain texture will be
        # resized in apply() if needed to match the output dimensions.
        # Using main_texture resolution would cause misalignment when the button
        # callback passes source.tensor but main_texture has different dimensions
        # (e.g., after crop effects).
        texture_height = current_image.shape[1]
        texture_width = current_image.shape[2]
        print(
            f"Texture dimensions from current_image: {texture_width}x{texture_height}, Density: {self.density}"
        )

        # Include image content hash in cache key when luma modulation is active,
        # so different images produce different grain textures.
        content_hash = None
        if self.luma_size_scale != 0.0:
            content_hash = self._compute_content_hash(current_image)

        # Create params dict for cache key
        cache_params = self._build_cache_params(
            texture_width, texture_height, content_hash
        )

        # Check if we already have this texture cached
        expected_path = TextureDiskCache.get_cache_path("GaussianGrain", cache_params)

        # If we have a grain_texture_path from deserialization, validate it matches current params
        if self.grain_texture_path is not None:
            if self.grain_texture_path != str(expected_path):
                print(
                    f"Cached path mismatch: expected {expected_path}, got {self.grain_texture_path}. Regenerating..."
                )
                self.grain_texture_path = None
                self.grain_texture = None

        if expected_path.exists():
            # Load from cache
            print(f"Loading grain texture from cache: {expected_path}")
            self.grain_texture = TextureDiskCache.load_texture(str(expected_path))
            self.grain_texture_path = str(expected_path)
        else:
            # Generate new texture
            num_grains = int(self.density * texture_width * texture_height)

            # Use Metal GPU acceleration (tiled) if available, otherwise fall back to CPU
            if METAL_AVAILABLE:
                print("Using Metal GPU acceleration (tiled) for grain generation")
                texture = generate_grain_texture_metal_tiled(
                    width=texture_width,
                    height=texture_height,
                    n_grains=num_grains,
                    size_mean=self.size_mean,
                    size_std=self.size_std,
                    intensity_mean=self.intensity_mean,
                    intensity_std=self.intensity_std,
                    color_shift=self.color_shift,
                    luminance_map=luminance_map,
                    luma_size_scale=self.luma_size_scale,
                    seed=self.seed,
                )
            else:
                print("Using CPU implementation for grain generation")
                texture = generate_grain_texture(
                    width=texture_width,
                    height=texture_height,
                    n_grains=num_grains,
                    size_mean=self.size_mean,
                    size_std=self.size_std,
                    intensity_mean=self.intensity_mean,
                    intensity_std=self.intensity_std,
                    color_shift=self.color_shift,
                    luma_size_scale=self.luma_size_scale,
                    seed=self.seed,
                    luminance_map=luminance_map,
                )

            # Convert to tensor
            self.grain_texture = torch.from_numpy(texture).permute(2, 0, 1)

            # Save to disk cache
            self.grain_texture_path = TextureDiskCache.save_texture(
                self.grain_texture, "GaussianGrain", cache_params
            )
            print(f"Saved grain texture to cache: {self.grain_texture_path}")

        self._texture_resolution = (texture_width, texture_height)
        self._built_with_params = self._current_grain_params()

    def adjust_parameters_for_preview(
        self, full_resolution: tuple[int, int], preview_resolution: tuple[int, int]
    ):
        """Scale grain strength to compensate for lower preview resolution."""
        full_h, full_w = full_resolution
        prev_h, prev_w = preview_resolution
        ratio = max(prev_h, prev_w) / max(full_h, full_w)
        if ratio < 1.0:
            self.strength *= ratio**0.5

    def _needs_rebuild(self, width: int, height: int) -> bool:
        """Check if grain texture needs to be rebuilt for new resolution."""
        if self._texture_resolution is None:
            return True
        tex_w, tex_h = self._texture_resolution
        # Rebuild if current image is larger than texture was built for
        return width > tex_w or height > tex_h

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        current_w, current_h = x.shape[2], x.shape[1]

        # Check if parameters changed since last build
        self._invalidate_if_params_changed()

        # Lazy load texture from disk if we have a path but no texture in memory
        if self.grain_texture is None and self.grain_texture_path is not None:
            self.grain_texture = TextureDiskCache.load_texture(self.grain_texture_path)

        # Rebuild if resolution increased (e.g., switching from preview to full res)
        if self.grain_texture is not None and self._needs_rebuild(current_w, current_h):
            print(
                f"Resolution increased from {self._texture_resolution} to ({current_w}, {current_h}), rebuilding grain texture"
            )
            self.grain_texture = None
            self.grain_texture_path = None

        # Build grain texture on first apply if not already built
        if self.grain_texture is None:
            self.build_grain_texture(x)

        if self.grain_texture is None:
            # If still None after build attempt, return unchanged
            print("WARNING: No grain texture available, returning image unchanged")
            return x
        else:
            grain_texture = self.grain_texture.to(x)
            if grain_texture.shape != x.shape:
                grain_texture = F.interpolate(
                    grain_texture.unsqueeze(0), size=x.shape[-2:], mode="bilinear"
                ).squeeze(0)

            # Multiplicative blending: modulate each color channel by grain
            # grain_texture is centered around 0, so (1 + strength * grain) creates variation
            # This preserves color variation in the grain (controlled by color_shift parameter)
            return x * (1.0 + self.strength * grain_texture)

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        shader = """
        vec4 apply_gaussiangrain(vec4 color, sampler2D grain_texture_sampler, float strength) {
            // Get proper UV coordinates from fragment position
            vec2 texSize = vec2(textureSize(main_texture, 0));
            vec2 uv = gl_FragCoord.xy / texSize;

            // Sample grain texture and apply multiplicative blending per channel
            // This preserves color variation in the grain (controlled by color_shift parameter)
            vec3 grain = texture(grain_texture_sampler, uv).rgb;
            return vec4(color.rgb * (1.0 + strength * grain), color.a);
        }
        """

        # Lazy load texture from disk if we have a path but no texture in memory
        if self.grain_texture is None and self.grain_texture_path is not None:
            self.grain_texture = TextureDiskCache.load_texture(self.grain_texture_path)

        # Get current image to check resolution, validate grain, and use for building if needed
        current_image = EffectStateCache.get_input_for_effect(self)
        if current_image is None:
            current_image = EffectStateCache.get_current_source()
        if current_image is None:
            try:
                current_image = TextureManager.download_texture("main_texture")
            except (ValueError, KeyError):
                return [shader], {
                    "u_grain_texture": torch.zeros((3, 512, 512)),
                    "u_strength": self.strength,
                }

        current_w, current_h = current_image.shape[2], current_image.shape[1]

        # Check if parameters changed since last build
        self._invalidate_if_params_changed()

        # Rebuild if resolution increased (e.g., switching from preview to full res)
        if self.grain_texture is not None and self._needs_rebuild(current_w, current_h):
            print(
                f"Resolution increased from {self._texture_resolution} to ({current_w}, {current_h}), rebuilding grain texture"
            )
            self.grain_texture = None
            self.grain_texture_path = None

        if self.grain_texture is None:
            self.build_grain_texture(current_image)

        if self.grain_texture is None:
            grain_texture = torch.zeros((3, 512, 512), dtype=torch.float32)
        else:
            grain_texture = self.grain_texture

        # Our handling of large tensors will automatically be assigned to a texture
        uniforms = {
            "u_grain_texture": grain_texture,
            "u_strength": self.strength,
        }
        return [shader], uniforms

    def serialize_to_cache(self) -> Dict[str, Any]:
        """Override to include grain_texture_path."""
        state = super().serialize_to_cache()
        if self.grain_texture_path is not None:
            state["grain_texture_path"] = self.grain_texture_path
        return state

    @classmethod
    def deserialize_from_cache(cls, state: Dict[str, Any]) -> "GaussianGrain":
        """Override to restore grain_texture_path."""
        # Remove grain_texture_path from state before passing to constructor
        grain_texture_path = state.pop("grain_texture_path", None)

        effect = super().deserialize_from_cache(state)

        # Restore grain_texture_path after creation
        if grain_texture_path is not None:
            effect.grain_texture_path = grain_texture_path

        return effect
