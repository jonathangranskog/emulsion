"""Cache for sharing intermediate effect processing states."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.effects.base import ImageEffect

# Downsample factor for memory efficiency
DOWNSAMPLE_FACTOR = 2


def _downsample(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return tensor.detach().cpu()
    return (
        F.interpolate(
            tensor.unsqueeze(0),
            scale_factor=1.0 / factor,
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .detach()
        .cpu()
    )


class EffectStateCache:
    """Stores downsampled intermediate states during effect processing.

    Uses class-level state like TextureManager for global access.
    """

    _intermediate_states: Dict[int, torch.Tensor] = {}
    _final_output: Optional[torch.Tensor] = None
    _valid: bool = False
    _effect_ids: List[int] = []
    _source_shape: Optional[Tuple[int, ...]] = None
    _current_source: Optional[torch.Tensor] = None

    @classmethod
    def invalidate(cls):
        cls._valid = False
        cls._intermediate_states.clear()
        cls._final_output = None
        cls._effect_ids.clear()

    @classmethod
    def is_valid(cls) -> bool:
        return cls._valid

    @classmethod
    def is_valid_for_effects(cls, effects: List[ImageEffect]) -> bool:
        if not cls._valid:
            return False
        current_ids = [id(e) for e in effects if e.toggled]
        return current_ids == cls._effect_ids

    @classmethod
    def set_source_shape(cls, shape: Tuple[int, ...]):
        cls._source_shape = shape

    @classmethod
    def source_changed(cls, source: torch.Tensor) -> bool:
        if cls._source_shape is None:
            return True
        return tuple(source.shape) != cls._source_shape

    @classmethod
    def begin_update(cls, effects: List[ImageEffect]):
        cls._intermediate_states.clear()
        cls._effect_ids = [id(e) for e in effects if e.toggled]
        cls._valid = False

    @classmethod
    def set_intermediate_state(cls, effect: ImageEffect, tensor: torch.Tensor):
        cls._intermediate_states[id(effect)] = _downsample(tensor, DOWNSAMPLE_FACTOR)

    @classmethod
    def get_input_for_effect(cls, effect: ImageEffect) -> Optional[torch.Tensor]:
        return cls._intermediate_states.get(id(effect))

    @classmethod
    def has_effect(cls, effect: ImageEffect) -> bool:
        return id(effect) in cls._intermediate_states

    @classmethod
    def set_final_output(cls, output: torch.Tensor):
        cls._final_output = _downsample(output, DOWNSAMPLE_FACTOR)
        cls._valid = True

    @classmethod
    def get_final_output(cls) -> Optional[torch.Tensor]:
        return cls._final_output if cls._valid else None

    @classmethod
    def set_current_source(cls, source: Optional[torch.Tensor]):
        """Set the current source tensor for effects that need it during get_shader_info."""
        cls._current_source = source

    @classmethod
    def get_current_source(cls) -> Optional[torch.Tensor]:
        """Get the current source tensor."""
        return cls._current_source

    @classmethod
    def clear(cls):
        cls._intermediate_states.clear()
        cls._final_output = None
        cls._valid = False
        cls._effect_ids.clear()
        cls._source_shape = None
        cls._current_source = None
