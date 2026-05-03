from dataclasses import dataclass, field

import torch

from src.core.device import get_preferred_device
from src.core.image import ImageData
from src.effects.crop import Crop
from src.pipeline.stack import EffectStack
from src.interface.texture import TextureManager


@dataclass
class TorchEffectsProcessor:
    source: ImageData
    stack: EffectStack
    _cache: torch.Tensor | None = None
    _bypass_dimensions: tuple[int, int] | None = None  # (W, H) when bypassed
    _needs_full_reprocess: bool = False

    def reconstruction_required(self) -> bool:
        return self.stack.reconstruction_required()

    def values_changed(self) -> bool:
        return self.stack.values_changed() or self._cache is None

    def get_output_texture_id(self) -> int:
        return TextureManager.get_texture_id("main_texture")

    def get_output_dimensions(self) -> tuple[int, int]:
        """Get the output dimensions (width, height) directly"""
        if self._bypass_dimensions is not None:
            return self._bypass_dimensions
        resolution = TextureManager.get_texture_resolution("main_texture")
        return (resolution[1], resolution[0])  # (W, H)

    def on_frame(self, texture):
        # After unbypass, trigger full reprocess if params changed during bypass
        if self._needs_full_reprocess and not self.stack.effects_bypassed:
            self._needs_full_reprocess = False
            self.stack.mark_values_changed()

        # When bypassed, only update bypass output if something changed
        if self.stack.effects_bypassed:
            if self.reconstruction_required() or self.values_changed():
                self._compute_bypass_tensor(texture)
                self.stack.clear_values_changed()
                self.stack.clear_reconstruction_required()
                self._needs_full_reprocess = True
            return

        if self.reconstruction_required() or self.values_changed():
            edited_tensor = self.get_edited_tensor()
            TextureManager.upload_texture_tensor(edited_tensor, texture)

    def get_edited_tensor(self) -> torch.Tensor:
        # This fetches the tensor from the source image and applies the effects
        # and caches the result for the next time.
        if self.reconstruction_required() or self.values_changed():
            # Process in float32, preserving HDR values (may be >1.0 or <0.0)
            device = get_preferred_device()
            original_device = self.source.tensor.device
            self._cache = self.stack.apply(self.source.tensor.float().to(device))
            # No clamping - preserve HDR values for EDR display
            # Note: Values will be clamped automatically when saving to PNG
            self._cache = self._cache.contiguous().to(original_device)
            self.stack.clear_values_changed()
            self.stack.clear_reconstruction_required()
        return self._cache

    def prepare_bypass_output(self, texture: str):
        """Prepare bypass display: upload source + crop only to GPU."""
        self._compute_bypass_tensor(texture)

    def restore_from_bypass(self, texture: str):
        """Restore the processed result after bypass — just re-upload cached tensor."""
        self._bypass_dimensions = None
        if self._cache is not None:
            TextureManager.upload_texture_tensor(self._cache, texture)

    def _compute_bypass_tensor(self, texture: str):
        """Compute source with crop-only applied and upload to GPU."""
        result = self.source.tensor
        for e in self.stack.effects:
            if isinstance(e, Crop) and e.toggled:
                result = e.apply(result)
                break
        _, h, w = result.shape
        self._bypass_dimensions = (w, h)
        TextureManager.upload_texture_tensor(result, texture)

    def fragment_shader_string(self) -> str:
        return """
        #version 330 core
        out vec4 FragColor;
        in vec2 TexCoord;
        uniform sampler2D main_texture;

        void main() {
            // Load the RGB values from the texture
            vec4 color = texture(main_texture, vec2(TexCoord.x, 1.0 - TexCoord.y));
            // Apply all the effects here below
            FragColor = color;
        }
        """

    def upload_uniforms(self, shader_program):
        pass

    def upload_effect_textures(self):
        pass

    def read_output_as_tensor(self) -> torch.Tensor:
        """Get the processed tensor output for saving

        Returns:
            Tensor with shape (C, H, W) in range [0, 1]
        """
        return self.get_edited_tensor()
