import copy
import os
from typing import Any, Dict, List

import torch

from src.effects.base import ImageEffect
from src.utils.lut_utils import (
    read_cube_file,
    apply_lut as apply_lut_util,
    identity_lut,
)
from src.search.nearest_lut import NearestLutSearch


class AbstractLUT(ImageEffect):
    def __init__(self, strength: float = 1.0):
        super().__init__()
        self.strength = strength
        self.domain_min = [0.0, 0.0, 0.0]
        self.domain_max = [1.0, 1.0, 1.0]
        self.lut_tensor = identity_lut()

    def get_params(self) -> List[Dict[str, Any]]:
        return [
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

    def ready(self) -> bool:
        return self.strength >= 1e-6

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3, "Input tensor must be 3D (C, H, W)"
        assert x.shape[0] == 3, "Input tensor must have 3 channels"
        if not self.ready():
            return x
        lut_applied = apply_lut_util(
            x, self.lut_tensor, self.domain_min, self.domain_max
        )
        return self.strength * lut_applied + (1 - self.strength) * x

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        glsl_code = """
        vec4 apply_lut(vec4 color, vec3 domain_max, vec3 domain_min, float strength, sampler3D lut) {
            // Apply domain scaling (fix: parentheses around subtraction)
            vec3 domain_space = (color.bgr - domain_min) / (domain_max - domain_min);

            // Clamp coordinates only for sampling, then apply delta to unclamped input
            vec3 clamped_coords = clamp(domain_space, 0.0, 1.0);

            // Sample LUT and mix
            vec3 lut_sampled = texture(lut, clamped_coords).rgb;
            vec4 result = vec4(lut_sampled, color.a);
            return mix(color, result, strength);
        }
        """

        uniforms = {
            "u_lut_strength": self.strength,
            "u_lut_tensor": self.lut_tensor,
            "u_domain_min": self.domain_min,
            "u_domain_max": self.domain_max,
        }

        return [glsl_code], uniforms

    def reset(self):
        self.lut_tensor = identity_lut()
        self.domain_min = [0.0, 0.0, 0.0]
        self.domain_max = [1.0, 1.0, 1.0]
        self.strength = 1.0

    def sync_unexposed_parameters(self, other: "AbstractLUT") -> bool:
        """Sync LUT tensor and domain from another LUT effect."""
        changed = False
        if not torch.equal(self.lut_tensor, other.lut_tensor):
            self.lut_tensor = other.lut_tensor.clone()
            changed = True
        if self.domain_min != other.domain_min:
            self.domain_min = list(other.domain_min)
            changed = True
        if self.domain_max != other.domain_max:
            self.domain_max = list(other.domain_max)
            changed = True
        return changed


class LUT(AbstractLUT):
    def __init__(self, lut_path: str = "", strength: float = 1.0):
        """
        LUT effect that applies a 3D LUT to an image based on a LUT file.

        Args:
            lut_path: Path to the LUT file.
            strength: Strength of the LUT effect.
            lut: LUT tensor. If provided, it will be used instead of reading from file.
        """
        super().__init__()
        self.lut_path = lut_path

    def reset_file(self):
        self.reset()
        self.lut_path = ""

    def get_params(self) -> List[Dict[str, Any]]:
        params = super().get_params()
        return [
            {
                "name": "lut_path",
                "label": "LUT File",
                "type": "file",
                "default": "",
                "file_types": [("LUT Files", "*.cube")],
            },
        ] + params

    def ready(self) -> bool:
        return super().ready() and self.lut_path != ""

    def on_file_load(self, lut_path: str):
        print(f"Loading LUT from {lut_path}")
        if not os.path.exists(lut_path):
            print(f"LUT file {lut_path} does not exist")
            return

        self.lut_path = lut_path
        self.lut_tensor, self.domain_min, self.domain_max = read_cube_file(lut_path)

    def serialize_to_cache(self) -> Dict[str, Any]:
        """
        Serialize LUT state including the loaded LUT data.

        Overrides base implementation to include lut_tensor, domain_min,
        domain_max to preserve the loaded LUT when restoring from cache.
        """
        state = super().serialize_to_cache()
        # Store the LUT data to preserve the loaded LUT
        state["lut_path"] = self.lut_path
        return state

    @classmethod
    def deserialize_from_cache(cls, state: Dict[str, Any]) -> "LUT":
        """
        Deserialize LUT from cached state.

        Overrides base implementation to properly restore the loaded LUT data.
        """
        # Create effect with lut_path and strength
        lut_path = state.get("lut_path", "")
        strength = state.get("strength", 1.0)
        effect = cls(lut_path=lut_path, strength=strength)

        # Restore the LUT data
        if lut_path != "":
            effect.on_file_load(lut_path)
        return effect


class SearchLUT(AbstractLUT):
    _shared_lut_search = None

    @classmethod
    def _get_lut_search(cls):
        if cls._shared_lut_search is None:
            cls._shared_lut_search = NearestLutSearch()
        return cls._shared_lut_search

    def __init__(self, strength: float = 1.0, search_query: str = ""):
        super().__init__(strength)
        self.search_query = search_query
        self.previous_search_query = ""
        self.first_search_query = search_query

    @property
    def nearest_lut_search(self):
        return SearchLUT._get_lut_search()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def get_params(self) -> List[Dict[str, Any]]:
        params = super().get_params()
        return [
            {
                "name": "search_query",
                "label": "Search Query",
                "type": "text",
                "default": self.first_search_query,
            },
        ] + params

    def ready(self) -> bool:
        return super().ready() and self.search_query != ""

    def update_lut(self):
        if self.ready() and self.search_query != self.previous_search_query:
            self.previous_search_query = self.search_query
            lut_data = self.nearest_lut_search.search(self.search_query)
            self.lut_tensor = torch.from_numpy(lut_data["lut_tensor"])
            self.domain_min = lut_data["domain_min"]
            self.domain_max = lut_data["domain_max"]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        self.update_lut()
        return super().apply(x)

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        self.update_lut()
        glsl_code, uniforms = super().get_shader_info()
        glsl_code[0] = glsl_code[0].replace("apply_lut", "apply_searchlut")
        return glsl_code, uniforms

    def sync_unexposed_parameters(self, other: "SearchLUT") -> bool:
        """Sync LUT state and search query from another SearchLUT effect."""
        changed = super().sync_unexposed_parameters(other)
        if self.previous_search_query != other.previous_search_query:
            self.previous_search_query = other.previous_search_query
            changed = True
        return changed

    def serialize_to_cache(self) -> Dict[str, Any]:
        """
        Serialize SearchLUT state including the selected LUT data.

        Overrides base implementation to include lut_tensor, domain_min,
        domain_max, and previous_search_query to preserve the exact LUT
        that was selected (not just the search query).
        """
        state = super().serialize_to_cache()
        # Store the LUT data to preserve the exact selected LUT
        state["lut_tensor"] = self.lut_tensor.cpu().numpy().tolist()
        state["domain_min"] = self.domain_min
        state["domain_max"] = self.domain_max
        state["previous_search_query"] = self.previous_search_query
        return state

    @classmethod
    def deserialize_from_cache(cls, state: Dict[str, Any]) -> "SearchLUT":
        """
        Deserialize SearchLUT from cached state.

        Overrides base implementation to properly restore the selected LUT data.
        """
        # Create effect with search query
        search_query = state.get("search_query", "")
        strength = state.get("strength", 1.0)
        effect = cls(strength=strength, search_query=search_query)

        # Restore the LUT data
        if "lut_tensor" in state:
            effect.lut_tensor = torch.tensor(state["lut_tensor"])
        if "domain_min" in state:
            effect.domain_min = state["domain_min"]
        if "domain_max" in state:
            effect.domain_max = state["domain_max"]
        if "previous_search_query" in state:
            effect.previous_search_query = state["previous_search_query"]

        return effect
