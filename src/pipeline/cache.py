import uuid
import os
import numpy as np
import torch
import json

import importlib

from src.pipeline.stack import EffectStack


class EffectsCache:
    """
    This is used to store the effects parameters when opening and closing the editor.

    When it is being deleted, it will save the current stack of effects and their parameters
    into a file, and then
    """

    def __init__(self, image_path: str, effects_stack: EffectStack):
        # Store a reference to the effects stack and the image path
        self.image_path = image_path
        self.uuid = uuid.uuid5(uuid.NAMESPACE_URL, image_path)
        self.effects_stack = effects_stack

    @property
    def cache_path(self):
        cache_path = os.path.join(".cache", f"{self.uuid}.json")
        return cache_path

    def _load_from_file(self, cache_path: str, exclude: set[str] | None = None) -> bool:
        if not os.path.exists(cache_path):
            return False
        try:
            print(f"Loading effects stack from cache file: {cache_path}")
            with open(cache_path, "r") as f:
                cache = json.load(f)
            effects_list = []
            for effect_data in cache:
                if exclude and effect_data.get("effect") in exclude:
                    print(
                        f"Skipping effect {effect_data.get('effect')} (excluded for seed load)"
                    )
                    continue
                try:
                    module = importlib.import_module(effect_data["module"])
                    effect_class = getattr(module, effect_data["effect"])
                    effects_list.append(
                        effect_class.deserialize_from_cache(effect_data["state"])
                    )
                except (ImportError, AttributeError, TypeError, KeyError) as e:
                    print(
                        f"Failed to load effect {effect_data.get('effect', 'unknown')}: {e}"
                    )
            self.effects_stack.set_effects(effects_list)
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"Failed to load cache: {e}")
            return False

    def load_from_cache(self) -> bool:
        return self._load_from_file(self.cache_path)

    def load_from_seed(self, seed_image_path: str) -> bool:
        """Load effects from another image's cache as the starting point."""
        seed_uuid = uuid.uuid5(uuid.NAMESPACE_URL, seed_image_path)
        seed_cache_path = os.path.join(".cache", f"{seed_uuid}.json")
        print(f"Seeding effects from: {seed_cache_path}")
        return self._load_from_file(seed_cache_path, exclude={"Crop"})

    def save_to_cache(self):
        try:
            os.makedirs(".cache", exist_ok=True)
            cache = []

            # Serialize each effect using its serialize_to_cache method
            for eff in self.effects_stack.effects:
                data = {
                    "effect": eff.__class__.__name__,
                    "module": eff.__class__.__module__,
                    "state": eff.serialize_to_cache(),
                }
                cache.append(data)

            # Save the data
            with open(self.cache_path, "w") as f:
                json.dump(cache, f, indent=4)
        except (IOError, OSError) as e:
            print(f"Failed to save cache: {e}")

    def __del__(self):
        try:
            self.save_to_cache()
        except Exception as e:
            # Silently fail or log - don't let exceptions escape __del__
            print(f"Error saving cache during cleanup: {e}")
