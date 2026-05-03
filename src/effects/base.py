from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch


class ImageEffect(ABC):
    """
    Abstract base class for effects that can be applied to a tensor
    """

    def __init__(self):
        self.toggled = True
        self.seed = 42

    # This one processes the tensor and returns the result
    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def get_params(self) -> List[Dict[str, Any]]: ...

    # Below is the code for the GPU-based effects

    # This one returns the shader code for the effect
    # including the dict of uniform names and values.
    # Values will be used to infer the type of the uniform too.
    @abstractmethod
    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]: ...

    def get_glsl_prefix(self) -> str:
        return self.__class__.__name__.lower()

    def reset_file(self):
        pass

    def vary_seed(self, offset: int):
        """Vary the seed by adding an offset."""
        self.seed = self.seed + offset

    def sync_unexposed_parameters(self, other: "ImageEffect") -> bool:
        """Sync internal state not exposed through get_params(). Returns True if changed."""
        return False

    def custom_render(self, input_texture: int, input_width: int, input_height: int):
        pass

    def get_effect_dimension_deltas(self) -> tuple[int, int]:
        return (0, 0)

    def requires_original_texture(self) -> bool:
        return False

    def adjust_parameters_for_preview(
        self, full_resolution: tuple[int, int], preview_resolution: tuple[int, int]
    ):
        """Adjust effect parameters in-place to compensate for lower preview resolution.

        Called on a deep-copied effect before applying it to a preview thumbnail.
        Override in subclasses where the visual result is resolution-dependent.

        Args:
            full_resolution: (height, width) of the original full-resolution image.
            preview_resolution: (height, width) of the preview thumbnail.
        """
        pass

    @property
    def requires_intermediate_state(self) -> bool:
        """Override to True if effect needs its input cached (e.g., for luminance)."""
        return False

    def poll_generation_status(self) -> Dict[str, Any] | None:
        """Poll asynchronous generation progress.

        Override in subclasses that perform background generation (e.g. T2S).
        Return a dict with at least a ``"status"`` key (``"generating"``,
        ``"success"``, or ``"error"``).  On error the dict should also
        contain ``"message"``.  Return ``None`` (the default) if the effect
        has no generation step.
        """
        return None

    def serialize_to_cache(self) -> Dict[str, Any]:
        """
        Serialize effect state for caching/undo.

        Default implementation serializes all parameters from get_params().
        Effects with complex state can override this method.

        Returns:
            Dictionary with serializable state
        """
        state = {}
        for param in self.get_params():
            if param["type"] != "button":
                value = getattr(self, param["name"])
                # Convert tensors to lists for JSON serialization
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy().tolist()
                state[param["name"]] = value
        return state

    @classmethod
    def deserialize_from_cache(cls, state: Dict[str, Any]) -> "ImageEffect":
        """
        Deserialize effect from cached state.

        Default implementation creates effect with no args and sets attributes.
        Effects with complex state can override this method.

        Args:
            state: Dictionary from serialize_to_cache()

        Returns:
            Reconstructed effect instance
        """
        effect = cls()
        for param_name, value in state.items():
            # Convert lists back to tensors
            if isinstance(value, list):
                value = torch.tensor(value)
            setattr(effect, param_name, value)
        return effect
