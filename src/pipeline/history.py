"""History management for effect stack undo/redo functionality."""

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EffectStackSnapshot:
    """Immutable snapshot of the entire effects stack state."""

    effects: List[Dict[str, Any]]
    timestamp: float
    description: str

    def __post_init__(self):
        """Ensure effects list is immutable by creating a deep copy."""
        import copy

        self.effects = copy.deepcopy(self.effects)


class EffectStackHistory:
    """Manages undo/redo history for an effects stack."""

    def __init__(self, max_history: int = 10):
        """
        Initialize history manager.

        Args:
            max_history: Maximum number of states to keep in history
        """
        self.history: List[EffectStackSnapshot] = []
        self.current_index: int = -1
        self.max_history = max_history

    def reset(self):
        self.history = []
        self.current_index = -1

    def capture_snapshot(self, effects: List[Any], description: str) -> None:
        """
        Capture current state of the effects stack.

        Args:
            effects: List of ImageEffect instances to serialize
            description: Human-readable description of the change
        """
        # Serialize the current effects list
        serialized = self._serialize_effects(effects)

        # Create snapshot
        snapshot = EffectStackSnapshot(
            effects=serialized, timestamp=time.time(), description=description
        )

        # If we're not at the end of history, truncate future states
        if self.current_index < len(self.history) - 1:
            self.history = self.history[: self.current_index + 1]

        # Add new snapshot
        self.history.append(snapshot)
        self.current_index = len(self.history) - 1

        # Enforce max history limit (circular buffer)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.current_index -= 1

    def undo(self) -> Optional[EffectStackSnapshot]:
        """
        Move back one state in history.

        Returns:
            Previous snapshot if available, None otherwise
        """
        if not self.can_undo():
            return None

        self.current_index -= 1
        return self.history[self.current_index]

    def redo(self) -> Optional[EffectStackSnapshot]:
        """
        Move forward one state in history.

        Returns:
            Next snapshot if available, None otherwise
        """
        if not self.can_redo():
            return None

        self.current_index += 1
        return self.history[self.current_index]

    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return self.current_index > 0

    def can_redo(self) -> bool:
        """Check if redo is possible."""
        return self.current_index < len(self.history) - 1

    def _serialize_effects(self, effects: List[Any]) -> List[Dict[str, Any]]:
        """
        Serialize effects list to dictionary representation.

        Args:
            effects: List of ImageEffect instances

        Returns:
            List of serialized effect dictionaries
        """
        serialized = []

        for eff in effects:
            effect_dict = {
                "effect": type(eff).__name__,
                "module": type(eff).__module__,
                "state": eff.serialize_to_cache(),
            }
            serialized.append(effect_dict)

        return serialized

    def deserialize_effects(self, data: List[Dict[str, Any]]) -> List[Any]:
        """
        Deserialize effects list from dictionary representation.

        Args:
            data: List of serialized effect dictionaries

        Returns:
            List of reconstructed ImageEffect instances
        """
        effects = []

        for effect_data in data:
            try:
                # Import module and get effect class
                module = importlib.import_module(effect_data["module"])
                effect_class = getattr(module, effect_data["effect"])

                # Deserialize using the effect's deserialize_from_cache method
                effect = effect_class.deserialize_from_cache(effect_data["state"])

                effects.append(effect)
            except Exception as e:
                print(
                    f"Warning: Failed to deserialize effect {effect_data.get('effect')}: {e}"
                )
                import traceback

                traceback.print_exc()
                continue

        return effects
