import torch

from dataclasses import dataclass, field
from typing import List

from src.effects.base import ImageEffect
from src.pipeline.history import EffectStackHistory, EffectStackSnapshot
from src.pipeline.state_cache import EffectStateCache


@dataclass
class EffectStack:
    """
    A stack of ImageEffects that can be applied to an image.
    If the stack is dirty, then the effects should be re-applied
    to update the result.
    """

    effects: List[ImageEffect] = field(default_factory=list)
    _values_changed: bool = True
    _reconstruction_required: bool = True
    effects_bypassed: bool = False
    history: EffectStackHistory = field(default_factory=EffectStackHistory)

    def __post_init__(self):
        """Capture initial state for undo/redo."""
        self.history.capture_snapshot(self.effects, "Initial state")

    def set_effects(self, effects: List[ImageEffect]):
        self.effects = effects
        self._values_changed = True

        # If this is being called right after initialization (e.g., loading from cache),
        # replace the initial state instead of creating a new history entry
        if len(self.history.history) == 1 and self.history.current_index == 0:
            # Replace the initial empty state with the loaded state
            self.history.reset()
            self.history.capture_snapshot(effects, "Initial state")

    def reconstruction_required(self) -> bool:
        return self._reconstruction_required

    def mark_reconstruction_required(self):
        self._reconstruction_required = True

    def clear_reconstruction_required(self):
        self._reconstruction_required = False

    def mark_values_changed(self):
        self._values_changed = True

    def clear_values_changed(self):
        self._values_changed = False

    def values_changed(self) -> bool:
        return self._values_changed

    def toggle_effects_bypass(self) -> bool:
        """Toggle effects bypass mode. Returns new bypass state."""
        self.effects_bypassed = not self.effects_bypassed
        return self.effects_bypassed

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all effects, caching intermediate states for effects that need them."""
        EffectStateCache.begin_update(self.effects)
        EffectStateCache.set_source_shape(tuple(x.shape))

        y = x
        for e in self.effects:
            if e.toggled:
                if e.requires_intermediate_state:
                    EffectStateCache.set_intermediate_state(e, y)
                y = e.apply(y)

        EffectStateCache.set_final_output(y)
        return y

    def capture_state(self, description: str):
        """Save current state after making changes."""
        self.history.capture_snapshot(self.effects, description)

    def undo(self) -> bool:
        """
        Restore previous state.

        Returns:
            True if undo was successful, False if no history available
        """
        snapshot = self.history.undo()
        if snapshot:
            self._restore_from_snapshot(snapshot)
            return True
        return False

    def redo(self) -> bool:
        """
        Restore next state.

        Returns:
            True if redo was successful, False if no future state available
        """
        snapshot = self.history.redo()
        if snapshot:
            self._restore_from_snapshot(snapshot)
            return True
        return False

    def move_effect(self, idx: int, direction: int) -> bool:
        """
        Move effect at idx up (direction=-1) or down (direction=1).

        Returns True if the move was performed, False if out of bounds.
        """
        new_idx = idx + direction
        if 0 <= new_idx < len(self.effects):
            self.effects[idx], self.effects[new_idx] = (
                self.effects[new_idx],
                self.effects[idx],
            )
            self.mark_reconstruction_required()
            self.mark_values_changed()
            return True
        return False

    def _restore_from_snapshot(self, snapshot):
        """Restore effects stack from a snapshot."""
        self.effects = self.history.deserialize_effects(snapshot.effects)
        self.mark_reconstruction_required()
        self.mark_values_changed()
