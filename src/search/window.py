from imgui_bundle import imgui, portable_file_dialogs
import imgui.integrations.glfw  # Required for proper imgui context
import copy
import json
import os
import queue
import time
import uuid
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from src.pipeline.stack import EffectStack
from src.pipeline.state_cache import EffectStateCache
from src.search.registry import full_effect_registry, EFFECT_NICKNAMES
from src.effects.lut import SearchLUT
from src.effects.text_to_shader import TextToShaderEffect
from src.search.nearest_lut import NearestLutSearch
from src.search.effect_presets import EffectPresetRegistry, EffectPreset
from src.search.gemini_action_matcher import GeminiActionMatcher
from src.search.gemini_agent_editor import GeminiAgentEditor, AgentState
from src.search.chat_message import (
    ChatMessage,
    ChatMode,
    MessageSender,
    MessageContentType,
    PreviewItem,
)
from src.interface.texture import TextureManager
from src.utils.lut_utils import apply_lut
from src.generation.shader_cache import ShaderCacheManager

NUM_PREVIEW_TEXTURES = 5
_MAX_DESCRIPTION_WORDS = 8


def _truncate_words(text: str, max_words: int = _MAX_DESCRIPTION_WORDS) -> str:
    """Truncate text to at most *max_words* words, appending '...' if trimmed."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


# Colour constants for the horizontal chat interface (RGBA tuples)
_ERROR_COLOR = (1.0, 0.4, 0.4, 1.0)
_STATUS_COLOR = (0.4, 0.9, 0.5, 1.0)
_SYSTEM_TEXT_COLOR = (0.90, 0.90, 0.90, 1.0)
_MODE_LABEL_COLOR = (0.6, 0.6, 0.6, 1.0)


class ChatWindow:
    """Chat-style window for interacting with the effects stack."""

    def __init__(self, effects_stack: EffectStack, image_path: str | None = None):
        self.effects_stack = effects_stack
        self.input_text = ""
        self.effect_registry = full_effect_registry()
        self.lut_search = NearestLutSearch()

        # Chat history persistence
        if image_path:
            _chat_uuid = uuid.uuid5(uuid.NAMESPACE_URL, image_path)
            self._chat_history_path: str | None = os.path.join(
                ".cache", f"{_chat_uuid}_chat.json"
            )
        else:
            self._chat_history_path = None

        # Chat state
        self.chat_messages: list[ChatMessage] = []
        self.mode = ChatMode.HUMAN_AGENT

        # Active preview state (only one set of previews is interactive at a time)
        self._active_preview_msg: ChatMessage | None = None
        self._active_lut_results: list | None = None
        self._active_effect_variants: list | None = None
        self._active_effect_name: str | None = None
        self._active_search_query: str | None = None

        # Effect preset system
        self.preset_registry = EffectPresetRegistry(
            effect_registry=self.effect_registry
        )

        # Preview generation state (shared between LUT and effect previews)
        self.preview_base_thumbnail = None
        self.preview_base_full_res = None
        self.preview_base_aspect_ratio = 1.0
        self.preview_base_stack_state = None
        self.preview_base_processor_id = None
        self.preview_cache = {}
        self.preview_textures_registered = False

        # Gemini action matcher for natural language query classification
        self.action_matcher = GeminiActionMatcher(self.effect_registry)

        # T2S (Text-to-Shader) system
        self.shader_cache = ShaderCacheManager()
        self.generating_t2s = False

        # Agentic editing state
        self._agent: GeminiAgentEditor | None = None
        self._agent_state: AgentState = AgentState.IDLE
        self._agent_running: bool = False
        self.agent_max_reviews: int = 0
        self.agent_max_steps: int = 4
        self.model_choice: str = "flash"  # "auto", "flash", or "pro"
        self.agent_auto_select: bool = False  # auto-select via vision

        # Human-Agent mode state: tracks pending step previews
        self._human_agent_pending_step: dict | None = None
        self._human_agent_candidates: list | None = None
        self._human_agent_effect_variants: list | None = None
        self._active_modify_stack_index: int | None = None  # for modify-op previews

        # Horizontal layout state
        self._plan_bullets: list[str] = []  # concise plan bullet points
        self._plan_step_count: int = 0
        self._current_step_index: int = -1  # -1 = no step active
        self._current_operation: str = ""  # "add" or "modify"
        self._current_effect_name: str = ""  # effect being worked on
        self._status_text: str = ""  # brief status/error summary
        self._status_type: MessageContentType = MessageContentType.TEXT

        # Reference image state
        self._reference_image_path: str | None = None
        self._reference_image_tensor: torch.Tensor | None = None  # CHW float32 [0,1]
        self._reference_texture_registered: bool = False
        self._reference_texture_pending_delete: bool = False
        self._REFERENCE_TEXTURE_NAME = "reference_image_thumb"

        # Search history state
        self.search_history: list[str] = []
        self.history_index = -1
        self.max_history_size = 50

        self._load_settings()
        self._load_chat_history()

    # ------------------------------------------------------------------
    # Chat history persistence
    # ------------------------------------------------------------------

    _SETTINGS_PATH = os.path.join(".cache", "settings.json")

    def _load_settings(self) -> None:
        try:
            with open(self._SETTINGS_PATH, "r") as f:
                data = json.load(f)
            self.agent_max_reviews = max(
                0, min(int(data.get("agent_max_reviews", 0)), 2)
            )
            self.agent_max_steps = int(data.get("agent_max_steps", 4))
            raw_choice = data.get("model_choice", "flash")
            self.model_choice = (
                raw_choice if raw_choice in ("auto", "flash", "pro") else "flash"
            )
            self.agent_auto_select = bool(data.get("agent_auto_select", False))
        except (IOError, json.JSONDecodeError, KeyError):
            pass  # Use defaults

    def _save_settings(self) -> None:
        try:
            os.makedirs(".cache", exist_ok=True)
            with open(self._SETTINGS_PATH, "w") as f:
                json.dump(
                    {
                        "agent_max_reviews": self.agent_max_reviews,
                        "agent_max_steps": self.agent_max_steps,
                        "model_choice": self.model_choice,
                        "agent_auto_select": self.agent_auto_select,
                    },
                    f,
                    indent=4,
                )
        except (IOError, OSError) as e:
            print(f"Failed to save settings: {e}")

    def _load_chat_history(self) -> None:
        if not self._chat_history_path or not os.path.exists(self._chat_history_path):
            return
        try:
            with open(self._chat_history_path, "r") as f:
                data = json.load(f)
            for entry in data:
                self.chat_messages.append(
                    ChatMessage(
                        sender=MessageSender[entry["sender"]],
                        content_type=MessageContentType[entry["content_type"]],
                        text=entry["text"],
                        timestamp=entry.get("timestamp", 0.0),
                    )
                )
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"Failed to load chat history: {e}")

    def save_chat_history(self) -> None:
        if not self._chat_history_path:
            return
        try:
            os.makedirs(".cache", exist_ok=True)
            saveable = [
                m
                for m in self.chat_messages
                if m.content_type != MessageContentType.PREVIEW
            ]
            data = [
                {
                    "sender": m.sender.name,
                    "content_type": m.content_type.name,
                    "text": m.text,
                    "timestamp": m.timestamp,
                }
                for m in saveable
            ]
            with open(self._chat_history_path, "w") as f:
                json.dump(data, f, indent=4)
        except (IOError, OSError) as e:
            print(f"Failed to save chat history: {e}")

    def __del__(self) -> None:
        try:
            self._save_settings()
            self.save_chat_history()
        except Exception:
            pass  # builtins may be gone during interpreter shutdown

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def add_user_message(self, text: str):
        """Append a user message to the chat history."""
        self.chat_messages.append(
            ChatMessage(
                sender=MessageSender.USER,
                content_type=MessageContentType.TEXT,
                text=text,
            )
        )

    def add_system_message(
        self,
        text: str,
        content_type: MessageContentType = MessageContentType.TEXT,
        preview_items: list[PreviewItem] | None = None,
        preview_effect_name: str | None = None,
        preview_source: str | None = None,
        is_selectable: bool = False,
    ) -> ChatMessage:
        """Append a system message (optionally with preview items)."""
        msg = ChatMessage(
            sender=MessageSender.SYSTEM,
            content_type=content_type,
            text=text,
            preview_items=preview_items or [],
            preview_effect_name=preview_effect_name,
            preview_source=preview_source,
            is_selectable=is_selectable,
        )
        self.chat_messages.append(msg)
        return msg

    # ------------------------------------------------------------------
    # Preview base / thumbnail helpers (unchanged from SearchWindow)
    # ------------------------------------------------------------------

    def _create_thumbnail(
        self, image: torch.Tensor, max_size: int = 768
    ) -> torch.Tensor:
        """Create a thumbnail from the input image, preserving aspect ratio."""
        if image is None:
            return None

        C, H, W = image.shape
        aspect_ratio = W / H
        if W > H:
            new_w = max_size
            new_h = int(max_size / aspect_ratio)
        else:
            new_h = max_size
            new_w = int(max_size * aspect_ratio)

        thumbnail = F.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return torch.clamp(thumbnail, 0.0, 1.0)

    def _get_effects_stack_state(self) -> tuple:
        """Get a hashable representation of the current effects stack state."""
        return (
            len(self.effects_stack.effects),
            id(self.effects_stack.effects),
            tuple(id(effect) for effect in self.effects_stack.effects),
            self.effects_stack._values_changed
            or self.effects_stack._reconstruction_required,
        )

    def _update_preview_base(self, effects_processor):
        """Update the preview base thumbnail from current effects processor output."""
        if effects_processor is None:
            return

        current_state = self._get_effects_stack_state()
        current_processor_id = id(effects_processor)

        if (
            current_state != self.preview_base_stack_state
            or current_processor_id != self.preview_base_processor_id
            or self.preview_base_thumbnail is None
        ):
            current_output = effects_processor.read_output_as_tensor()
            self.preview_base_full_res = (
                current_output.shape[1],
                current_output.shape[2],
            )
            self.preview_base_thumbnail = self._create_thumbnail(current_output)

            if self.preview_base_thumbnail is not None:
                _, H, W = self.preview_base_thumbnail.shape
                self.preview_base_aspect_ratio = W / H

            self.preview_base_stack_state = current_state
            self.preview_base_processor_id = current_processor_id
            self.preview_textures_registered = False

    # ------------------------------------------------------------------
    # Search / preview state management
    # ------------------------------------------------------------------

    def _clear_active_previews(self):
        """Deactivate the currently interactive preview set."""
        if self._active_preview_msg is not None:
            self._active_preview_msg.is_selectable = False
        self._active_preview_msg = None
        self._active_lut_results = None
        self._active_effect_variants = None
        self._active_effect_name = None
        self._active_search_query = None
        self.preview_textures_registered = False
        self._active_modify_stack_index = None
        # Force the preview base to be re-read from the effects processor
        # on the next texture generation.  Without this, "modify" operations
        # can leave the cached state unchanged (same effect count / ids,
        # _values_changed toggled back to False), causing _update_preview_base
        # to skip the re-read and produce stale preview thumbnails.
        self.preview_base_stack_state = None

    def _add_to_history(self, query: str):
        """Add a search query to the history (avoids consecutive duplicates)."""
        query = query.strip()
        if not query:
            return
        if self.search_history and self.search_history[-1] == query:
            return
        self.search_history.append(query)
        if len(self.search_history) > self.max_history_size:
            self.search_history.pop(0)

    # ------------------------------------------------------------------
    # Reference image management
    # ------------------------------------------------------------------

    def _load_reference_image(self) -> None:
        """Open a file dialog and load a reference image."""
        dialog = portable_file_dialogs.open_file(
            title="Select Reference Image",
            default_path=".",
            filters=["Image files|*.jpg *.jpeg *.png *.tiff *.tif *.bmp"],
        )
        result = dialog.result()
        if not result or len(result) == 0:
            return
        filepath = result[0]
        try:
            img = Image.open(filepath).convert("RGB")
            tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            # Create a small thumbnail for display (max 64px on long edge)
            C, H, W = tensor.shape
            long_edge = max(H, W)
            thumb_size = 64
            if long_edge > thumb_size:
                scale = thumb_size / long_edge
                new_h = max(1, int(H * scale))
                new_w = max(1, int(W * scale))
                thumb = F.interpolate(
                    tensor.unsqueeze(0), size=(new_h, new_w),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
            else:
                thumb = tensor
            self._reference_image_path = filepath
            self._reference_image_tensor = tensor
            # Register thumbnail texture for display
            self._register_reference_texture(thumb)
        except Exception as e:
            self._set_status(f"Failed to load reference image: {e}", MessageContentType.ERROR)

    def _register_reference_texture(self, thumb: torch.Tensor) -> None:
        """Register (or update) the reference image thumbnail texture."""
        name = self._REFERENCE_TEXTURE_NAME
        if self._reference_texture_registered:
            try:
                TextureManager.delete_texture(name)
            except (ValueError, Exception):
                pass
            self._reference_texture_registered = False
        TextureManager.register_texture2d(thumb, name)
        self._reference_texture_registered = True

    def _remove_reference_image(self) -> None:
        """Remove the currently loaded reference image.

        The GPU texture is not deleted immediately because ImGui may still
        reference it in the current frame's draw list.  Instead we flag it
        for deletion at the start of the next render() call.
        """
        self._reference_image_path = None
        self._reference_image_tensor = None
        if self._reference_texture_registered:
            self._reference_texture_pending_delete = True

    # ------------------------------------------------------------------
    # Preview texture generation
    # ------------------------------------------------------------------

    def _ensure_preview_textures(self, effects_processor, items, generator_fn):
        """Generate and register preview textures if not already done."""
        self._update_preview_base(effects_processor)

        if self.preview_textures_registered or self.preview_base_thumbnail is None:
            return

        # For "modify" operations, build a preview base that excludes the
        # effect being modified so candidates aren't double-applied.
        preview_base = self.preview_base_thumbnail
        if self._active_modify_stack_index is not None and effects_processor is not None:
            modify_base = self._build_modify_preview_base(effects_processor)
            if modify_base is not None:
                preview_base = modify_base

        self.preview_cache.clear()
        for i, item in enumerate(items):
            assert i < NUM_PREVIEW_TEXTURES, (
                f"Too many items ({len(items)} > {NUM_PREVIEW_TEXTURES})"
            )
            texture_name = f"preview_{i}"
            preview_tensor = generator_fn(item, i, preview_base)
            preview_tensor = torch.clamp(preview_tensor, 0.0, 1.0)
            self.preview_cache[texture_name] = preview_tensor
            TextureManager.register_texture2d(preview_tensor, texture_name)

        self.preview_textures_registered = True

    def _build_modify_preview_base(self, effects_processor) -> torch.Tensor | None:
        """Build a preview base that excludes the effect at _active_modify_stack_index.

        For "modify" operations the normal preview base already contains the
        existing effect.  To show accurate previews we need to re-apply the
        effect stack on the source thumbnail while skipping the target effect.
        """
        skip_idx = self._active_modify_stack_index
        effects = self.effects_stack.effects
        if skip_idx is None or not (0 <= skip_idx < len(effects)):
            return None

        try:
            source_tensor = effects_processor.source.tensor.float()
            thumb = self._create_thumbnail(source_tensor)
            if thumb is None:
                return None

            _, th, tw = thumb.shape
            full_res = self.preview_base_full_res or (th, tw)

            for i, effect in enumerate(effects):
                if i == skip_idx or not effect.toggled:
                    continue
                eff_copy = copy.deepcopy(effect)
                eff_copy.adjust_parameters_for_preview(full_res, (th, tw))
                thumb = eff_copy.apply(thumb)

            return torch.clamp(thumb, 0.0, 1.0)
        except Exception:
            return None

    def _lut_preview_generator(self, lut_result, index, preview_base):
        """Generate preview by applying LUT to preview base."""
        lut_tensor = lut_result["lut_tensor"]
        if isinstance(lut_tensor, np.ndarray):
            lut_tensor = torch.from_numpy(lut_tensor)
        domain_min = lut_result.get("domain_min", [0.0, 0.0, 0.0])
        domain_max = lut_result.get("domain_max", [1.0, 1.0, 1.0])
        return apply_lut(preview_base, lut_tensor, domain_min, domain_max)

    def _effect_preview_generator(self, effect_variant, index, preview_base):
        """Generate preview by applying an effect variant to preview base."""
        effect_copy = copy.deepcopy(effect_variant)
        if self.preview_base_full_res is not None:
            _, prev_h, prev_w = preview_base.shape
            effect_copy.adjust_parameters_for_preview(
                self.preview_base_full_res, (prev_h, prev_w)
            )
        try:
            return effect_copy.apply(preview_base.clone())
        except Exception:
            return preview_base.clone()

    def _handle_preview_selection(
        self, msg: ChatMessage, pitem: PreviewItem, index: int
    ):
        """Handle a preview image click."""
        # Route to Human-Agent handler if we're awaiting selection
        if self._agent_state == AgentState.AWAITING_HUMAN_SELECTION:
            self._handle_human_agent_preview_selection(msg, pitem, index)
            return

        if msg.preview_source == "lut" and self._active_lut_results:
            selected_lut = self._active_lut_results[index]
            query = self._active_search_query or ""
            new_effect = SearchLUT(search_query=query)
            new_effect.lut_tensor = torch.from_numpy(selected_lut["lut_tensor"])
            new_effect.domain_min = selected_lut["domain_min"]
            new_effect.domain_max = selected_lut["domain_max"]
            new_effect.previous_search_query = query

            self.effects_stack.effects.append(new_effect)
            self.reconstruction_required(f"Add LUT {selected_lut['lut_name']}")
            self._set_status(
                f"Applied LUT: {selected_lut['lut_name']}",
                MessageContentType.STATUS,
            )
            self._clear_active_previews()

        elif msg.preview_source == "effect" and self._active_effect_variants:
            selected_variant = self._active_effect_variants[index]
            effect_name = self._active_effect_name or "effect"
            self.effects_stack.effects.append(selected_variant)
            self.reconstruction_required(f"Add {effect_name}")
            self._set_status(
                f"Applied '{effect_name}': {pitem.label}",
                MessageContentType.STATUS,
            )
            self._clear_active_previews()

    # ------------------------------------------------------------------
    # History callback
    # ------------------------------------------------------------------

    def _history_callback(self, data):
        """Callback for arrow key history navigation in input text."""
        if data.event_key == 3:  # Up Arrow
            if self.search_history:
                if self.history_index == -1:
                    self.history_index = len(self.search_history) - 1
                elif self.history_index > 0:
                    self.history_index -= 1
                history_text = self.search_history[self.history_index]
                data.delete_chars(0, data.cursor_pos)
                data.insert_chars(0, history_text)

        elif data.event_key == 4:  # Down Arrow
            if self.history_index >= 0:
                if self.history_index < len(self.search_history) - 1:
                    self.history_index += 1
                    history_text = self.search_history[self.history_index]
                    data.delete_chars(0, data.cursor_pos)
                    data.insert_chars(0, history_text)
                else:
                    data.delete_chars(0, data.cursor_pos)
                    self.history_index = -1

        return 0

    # ------------------------------------------------------------------
    # Action processing (Enter pressed)
    # ------------------------------------------------------------------

    def _try_shortcut(self, query: str, effects_processor) -> bool:
        """Try to handle query via fast-path shortcuts without planning.

        Returns True if the query was handled, False to fall through to
        the Human-Agent planning pipeline.
        """
        lower = query.lower().strip()

        # --- Exact stack commands ---
        if lower == "undo":
            self.add_user_message(query)
            if self.effects_stack.undo():
                self._set_status("Undo successful", MessageContentType.STATUS)
            else:
                self._set_status("Nothing to undo", MessageContentType.ERROR)
            self._add_to_history(query)
            return True

        if lower == "redo":
            self.add_user_message(query)
            if self.effects_stack.redo():
                self._set_status("Redo successful", MessageContentType.STATUS)
            else:
                self._set_status("Nothing to redo", MessageContentType.ERROR)
            self._add_to_history(query)
            return True

        if lower == "clear":
            self.add_user_message(query)
            self.effects_stack.effects = []
            self.reconstruction_required("Clear all effects")
            self._set_status("Cleared effect stack", MessageContentType.STATUS)
            self._add_to_history(query)
            return True

        # --- Remove command ---
        if lower == "remove" or lower.startswith("remove "):
            self.add_user_message(query)
            rest = lower[6:].strip()
            effect_name = rest if rest in self.effect_registry else (rest or None)
            self._handle_remove(effect_name, query)
            return True

        # --- T2S prefix ---
        for prefix in ("t2s:", "text-to-shader:"):
            if lower.startswith(prefix):
                self.add_user_message(query)
                prompt = query[len(prefix):].strip()
                self._handle_t2s(prompt or None, query)
                return True

        # --- Search/LUT prefix (new) ---
        for prefix in ("search:", "lut:"):
            if lower.startswith(prefix):
                stripped = query[len(prefix):].strip()
                self.add_user_message(query)
                if not stripped:
                    self._set_status(
                        "Please provide a search term after 'search:'",
                        MessageContentType.ERROR,
                    )
                    return True
                self._clear_active_previews()
                lut_results = self.lut_search.search_top_k(stripped, k=5)
                self._active_lut_results = lut_results
                self._active_search_query = stripped
                self.preview_textures_registered = False
                preview_items = [
                    PreviewItem(
                        texture_name=f"preview_{i}",
                        label=f"Option {i + 1}: {r['lut_name']}",
                        item_data=r,
                        item_index=i,
                    )
                    for i, r in enumerate(lut_results)
                ]
                msg = self.add_system_message(
                    f"Found {len(lut_results)} LUT matches. Click to apply:",
                    MessageContentType.PREVIEW,
                    preview_items=preview_items,
                    preview_source="lut",
                    is_selectable=True,
                )
                self._active_preview_msg = msg
                self._add_to_history(query)
                return True

        # --- Exact effect name/nickname match ---
        if lower in self.effect_registry:
            self.add_user_message(query)
            self._clear_active_previews()
            action_result = {
                "action": "add",
                "effect_name": lower,
                "prompt": None,
                "presets": None,
            }
            self._handle_add(lower, action_result, query, effects_processor)
            return True

        # No shortcut matched
        return False

    def _handle_remove(self, effect_name, search_query):
        """Process a 'remove' action."""
        if effect_name and effect_name in self.effect_registry:
            removed = False
            for effect in reversed(self.effects_stack.effects):
                effect_class_name = effect.__class__.__name__
                effect_nicknames = EFFECT_NICKNAMES.get(effect_class_name, [])
                if (
                    effect_class_name.lower() == effect_name
                    or effect_name in effect_nicknames
                ):
                    self.effects_stack.effects.remove(effect)
                    self.reconstruction_required(f"Remove {effect_name}")
                    self._set_status(
                        f"Removed '{effect_name}'",
                        MessageContentType.STATUS,
                    )
                    removed = True
                    break
            if not removed:
                self._set_status(
                    f"No '{effect_name}' found in stack",
                    MessageContentType.ERROR,
                )
        else:
            if self.effects_stack.effects:
                self.effects_stack.effects.pop()
                self.reconstruction_required("Remove last effect")
                self._set_status(
                    "Removed last effect", MessageContentType.STATUS
                )
            else:
                self._set_status(
                    "No effects to remove", MessageContentType.ERROR
                )
        self._add_to_history(search_query)

    def _handle_add(self, effect_name, action_result, search_query, effects_processor):
        """Process an 'add' action (effect with presets, immediate add, or LUT search)."""
        if effect_name and effect_name in self.effect_registry:
            # Determine presets: prefer Gemini-generated, fall back to hardcoded
            gemini_presets = action_result.get("presets")
            if gemini_presets:
                presets = [
                    EffectPreset(label=p["label"], params=p["params"])
                    for p in gemini_presets
                ]
            elif self.preset_registry.has_presets(effect_name):
                presets = self.preset_registry.get_presets(effect_name)
            else:
                presets = None

            if presets:
                # Build effect variants for preview
                variants = []
                preview_items = []
                for i, preset in enumerate(presets):
                    effect_instance = self.effect_registry[effect_name]()
                    for param_name, param_value in preset.params.items():
                        setattr(effect_instance, param_name, param_value)
                    variants.append(effect_instance)
                    preview_items.append(
                        PreviewItem(
                            texture_name=f"preview_{i}",
                            label=preset.label,
                            item_data=effect_instance,
                            item_index=i,
                        )
                    )

                self._active_effect_variants = variants
                self._active_effect_name = effect_name
                self._active_search_query = search_query
                self.preview_textures_registered = False

                msg = self.add_system_message(
                    f"Select a '{effect_name}' variant:",
                    MessageContentType.PREVIEW,
                    preview_items=preview_items,
                    preview_effect_name=effect_name,
                    preview_source="effect",
                    is_selectable=True,
                )
                self._active_preview_msg = msg
                self._add_to_history(search_query)
            else:
                # No presets - immediate add
                new_effect = self.effect_registry[effect_name]()
                self.effects_stack.effects.append(new_effect)
                self.reconstruction_required(f"Add {effect_name}")
                self._set_status(
                    f"Added '{effect_name}'",
                    MessageContentType.STATUS,
                )
                self._add_to_history(search_query)
        else:
            # No effect match - search for LUTs
            lut_results = self.lut_search.search_top_k(search_query, k=5)
            self._active_lut_results = lut_results
            self._active_search_query = search_query
            self.preview_textures_registered = False

            preview_items = [
                PreviewItem(
                    texture_name=f"preview_{i}",
                    label=f"Option {i + 1}: {r['lut_name']}",
                    item_data=r,
                    item_index=i,
                )
                for i, r in enumerate(lut_results)
            ]

            msg = self.add_system_message(
                f"Found {len(lut_results)} LUT matches. Click to apply:",
                MessageContentType.PREVIEW,
                preview_items=preview_items,
                preview_source="lut",
                is_selectable=True,
            )
            self._active_preview_msg = msg
            self._add_to_history(search_query)

    def _handle_t2s(self, t2s_prompt, search_query):
        """Process a 't2s' (text-to-shader) action."""
        prompt = t2s_prompt or search_query
        for prefix in ["t2s:", "text-to-shader:"]:
            if prompt.lower().startswith(prefix):
                prompt = prompt[len(prefix) :].strip()

        if not prompt:
            self._set_status(
                "Please provide a description for the shader",
                MessageContentType.ERROR,
            )
        else:
            new_effect = TextToShaderEffect(
                prompt=prompt, cache_manager=self.shader_cache
            )
            self.effects_stack.effects.append(new_effect)
            self.reconstruction_required(f"Add T2S: {prompt[:30]}...")
            self._set_status(
                f"Generating shader: '{prompt[:40]}...'",
                MessageContentType.STATUS,
            )
            self.generating_t2s = True
            self._add_to_history(search_query)

    # ------------------------------------------------------------------
    # T2S status polling
    # ------------------------------------------------------------------

    def _poll_t2s_status(self):
        """Check T2S generation progress and append status messages."""
        if not self.generating_t2s:
            return
        for i, effect in enumerate(reversed(self.effects_stack.effects)):
            gen = effect.poll_generation_status()
            if gen is not None:
                if gen["status"] == "success":
                    self._set_status(
                        "Shader generated successfully!",
                        MessageContentType.STATUS,
                    )
                    self.generating_t2s = False
                elif gen["status"] == "error":
                    self._set_status(
                        f"Shader failed: {gen['message']}",
                        MessageContentType.ERROR,
                    )
                    self.generating_t2s = False
                    actual_index = len(self.effects_stack.effects) - 1 - i
                    self.effects_stack.effects.pop(actual_index)
                    self.reconstruction_required("Remove failed T2S effect")
                # "generating" — no new message, just wait
                break

    # ------------------------------------------------------------------
    # Agentic editing
    # ------------------------------------------------------------------

    def _resolve_model(self) -> str:
        """Return the Gemini model ID based on user's model_choice setting.

        Auto logic: Pro when max_steps > 2, Flash otherwise.
        """
        if self.model_choice == "flash":
            return "gemini-3.1-flash-lite-preview"
        if self.model_choice == "pro":
            return "gemini-3.1-pro-preview"
        # auto
        if self.agent_max_steps > 2:
            return "gemini-3.1-pro-preview"
        return "gemini-3.1-flash-lite-preview"

    def _start_human_agent_session(
        self, user_prompt: str, effects_processor
    ) -> None:
        """Launch a Human-Agent editing session from the given user prompt."""
        self._update_preview_base(effects_processor)
        if self.preview_base_thumbnail is None:
            self._set_status("No image loaded.", MessageContentType.ERROR)
            return

        vision_thumbnail = self._create_thumbnail(
            self.preview_base_thumbnail, max_size=256
        )

        stack_snapshot = [
            {"effect": type(e).__name__, "state": e.serialize_to_cache()}
            for e in self.effects_stack.effects
        ]

        if self._agent is None:
            self._agent = GeminiAgentEditor(
                effect_registry=self.effect_registry,
                preset_registry=self.preset_registry,
                params_catalog=self.action_matcher._params_catalog,
                effects_list=self.action_matcher._effects_list,
            )

        # Prepare reference image thumbnail for the agent (256px max)
        reference_thumbnail = None
        if self._reference_image_tensor is not None:
            reference_thumbnail = self._create_thumbnail(
                self._reference_image_tensor, max_size=256
            )

        self._agent.start(
            user_prompt,
            vision_thumbnail,
            stack_snapshot,
            max_reviews=self.agent_max_reviews,
            enable_refinement=False,
            max_steps=self.agent_max_steps,
            model=self._resolve_model(),
            human_selection=True,
            auto_select=self.agent_auto_select,
            reference_image=reference_thumbnail,
        )
        self._agent_running = True
        self._agent_state = AgentState.PLANNING
        # Reset horizontal layout state for new session
        self._plan_bullets.clear()
        self._current_step_index = -1
        self._current_effect_name = ""
        self._current_operation = ""
        self._set_status(
            f'Planning edits for: "{user_prompt}"',
            MessageContentType.STATUS,
        )

    def _poll_agent_status(self) -> None:
        """Drain the agent ui_queue and process messages on the main thread."""
        if not self._agent_running or self._agent is None:
            return

        # Process all pending messages without blocking
        while True:
            try:
                msg = self._agent.ui_queue.get_nowait()
            except queue.Empty:
                break

            msg_type = msg.get("type", "")

            if msg_type == "status":
                self._set_status(msg["text"], MessageContentType.STATUS)

            elif msg_type == "error":
                self._set_status(msg["text"], MessageContentType.ERROR)
                self._agent_running = False
                self._agent_state = AgentState.ERROR
                # Clear plan/step state so the user starts fresh
                self._plan_bullets.clear()
                self._current_step_index = -1
                self._current_effect_name = ""
                self._current_operation = ""
                self._clear_active_previews()
                self._human_agent_pending_step = None
                self._human_agent_candidates = None
                self._human_agent_effect_variants = None

            elif msg_type == "plan_ready":
                n = msg["step_count"]
                plan_steps = msg.get("steps", [])
                # Build concise bullet-point plan
                self._plan_bullets.clear()
                self._plan_step_count = n
                for i, s in enumerate(plan_steps):
                    op = s.get("operation", "add")
                    name = s.get("effect_name", "")
                    rationale = s.get("rationale", "")
                    verb = "Remove" if op == "remove" else ("Adjust" if op == "modify" else "Add")
                    bullet = f"  {i + 1}. {verb} {name}"
                    if rationale:
                        bullet += f" - {_truncate_words(rationale)}"
                    self._plan_bullets.append(bullet)
                # Also persist to chat history
                plan_text = _truncate_words(msg["goal_summary"])
                if plan_steps:
                    plan_text += "\n" + "\n".join(self._plan_bullets)
                else:
                    plan_text += f" ({n} step{'s' if n != 1 else ''})"
                self.add_system_message(plan_text, MessageContentType.STATUS)
                self._agent_state = AgentState.EXECUTING
                self._status_text = ""

            elif msg_type == "step_status":
                step_idx = msg["step_index"]
                effect_name = msg["effect_name"]
                self._current_step_index = step_idx
                self._current_effect_name = effect_name
                self._current_operation = msg.get("operation", "add")
                self._set_status(
                    f"Preparing {effect_name}...",
                    MessageContentType.STATUS,
                )

            elif msg_type == "apply_effect":
                self._apply_agent_effect(msg)
                self._agent.confirm_applied()

            elif msg_type == "review_plan":
                review_steps = msg.get("steps", [])
                critique = msg.get("critique", "")
                # Update plan bullets with review corrections
                self._plan_bullets.clear()
                self._plan_step_count = len(review_steps)
                self._current_step_index = -1
                for i, s in enumerate(review_steps):
                    op = s.get("operation", "add")
                    name = s.get("effect_name", "")
                    rationale = s.get("rationale", "")
                    verb = "Remove" if op == "remove" else ("Adjust" if op == "modify" else "Add")
                    bullet = f"  {i + 1}. {verb} {name}"
                    if rationale:
                        bullet += f" - {_truncate_words(rationale)}"
                    self._plan_bullets.append(bullet)
                # Persist to chat history
                review_text = f"Review: {_truncate_words(critique)}"
                if review_steps:
                    review_text += "\n" + "\n".join(self._plan_bullets)
                self.add_system_message(review_text, MessageContentType.STATUS)

            elif msg_type == "step_previews":
                self._handle_step_previews(msg)

            elif msg_type == "done":
                self._set_status(msg["summary"], MessageContentType.STATUS)
                self._current_step_index = -1
                self._current_effect_name = ""
                self._current_operation = ""
                self._agent_running = False
                self._agent_state = AgentState.AWAITING_USER

    def _apply_agent_effect(self, msg: dict) -> None:
        """Apply a single agent-chosen effect to the effects stack (main thread)."""
        operation = msg.get("operation", "add")
        effect_name = msg.get("effect_name", "")
        params = msg.get("params", {})
        stack_index = msg.get("stack_index")

        if operation == "add":
            if effect_name not in self.effect_registry:
                return
            new_effect = self.effect_registry[effect_name]()
            for pname, pval in params.items():
                setattr(new_effect, pname, pval)
            # Apply pre-loaded LUT data for search_lut effects
            lut_data = msg.get("lut_data")
            if lut_data is not None:
                new_effect.lut_tensor = lut_data["lut_tensor"]
                new_effect.domain_min = lut_data["domain_min"]
                new_effect.domain_max = lut_data["domain_max"]
                new_effect.previous_search_query = getattr(
                    new_effect, "search_query", ""
                )
            self.effects_stack.effects.append(new_effect)
            self.reconstruction_required(f"Agent add {effect_name}")

        elif operation == "modify":
            if stack_index is None or not (
                0 <= stack_index < len(self.effects_stack.effects)
            ):
                return
            effect = self.effects_stack.effects[stack_index]
            for pname, pval in params.items():
                setattr(effect, pname, pval)
            self.effects_stack.mark_values_changed()
            self.effects_stack.capture_state(f"Agent modify {effect_name}")

        elif operation == "remove":
            if stack_index is None or not (
                0 <= stack_index < len(self.effects_stack.effects)
            ):
                return
            self.effects_stack.effects.pop(stack_index)
            self.reconstruction_required(f"Agent remove {effect_name}")

    # ------------------------------------------------------------------
    # Human-Agent mode: step preview handling
    # ------------------------------------------------------------------

    def _handle_step_previews(self, msg: dict) -> None:
        """Handle a step_previews message from the agent in Human-Agent mode.

        Builds effect variants from the candidate data, sets up preview state,
        and shows selectable previews in the chat.
        """
        effect_name = msg.get("effect_name", "")
        candidates = msg.get("candidates", [])
        step_index = msg.get("step_index", 0)
        rationale = msg.get("rationale", "")
        operation = msg.get("operation", "add")

        # Update horizontal layout title state
        self._current_step_index = step_index
        self._current_effect_name = effect_name
        self._current_operation = operation

        if not candidates or effect_name not in self.effect_registry:
            # Skip — tell agent to move on
            self._agent.agent_queue.put({"action": "skip"})
            return

        # Build effect instances from candidate param dicts
        variants = []
        preview_items = []
        for i, cand in enumerate(candidates):
            eff = self.effect_registry[effect_name]()
            for pname, pval in cand.get("params", {}).items():
                setattr(eff, pname, pval)
            # For search_lut, apply pre-loaded LUT data so previews
            # show the exact matched LUT (not just the #1 search result).
            lut_data = cand.get("lut_data")
            if lut_data is not None:
                eff.lut_tensor = lut_data["lut_tensor"]
                eff.domain_min = lut_data["domain_min"]
                eff.domain_max = lut_data["domain_max"]
                eff.previous_search_query = getattr(eff, "search_query", "")
            variants.append(eff)
            preview_items.append(
                PreviewItem(
                    texture_name=f"preview_{i}",
                    label=cand.get("label", f"Option {i + 1}"),
                    item_data=eff,
                    item_index=i,
                )
            )

        # Store state for selection handling
        self._human_agent_pending_step = msg
        self._human_agent_candidates = candidates
        self._human_agent_effect_variants = variants

        # Set up preview textures
        self._clear_active_previews()
        self._active_effect_variants = variants
        self._active_effect_name = effect_name
        self.preview_textures_registered = False

        # For "modify" operations, record the stack index so the preview
        # generator can build a base image that excludes the existing effect.
        if operation == "modify" and msg.get("stack_index") is not None:
            self._active_modify_stack_index = msg["stack_index"]

        # Show preview message in chat (persisted, but title is rendered via _render_step_title)
        verb = "Adjusting" if operation == "modify" else "Adding"
        short_rationale = _truncate_words(rationale) if rationale else ""
        self._status_text = f"Select a variant ({short_rationale})" if short_rationale else "Select a variant"
        self._status_type = MessageContentType.TEXT
        preview_msg = self.add_system_message(
            f"Step {step_index + 1}: {verb} {effect_name} - {short_rationale}",
            MessageContentType.PREVIEW,
            preview_items=preview_items,
            preview_effect_name=effect_name,
            preview_source="effect",
            is_selectable=True,
        )
        self._active_preview_msg = preview_msg
        self._agent_state = AgentState.AWAITING_HUMAN_SELECTION

    def _handle_human_agent_preview_selection(
        self, msg: ChatMessage, pitem: PreviewItem, index: int
    ) -> None:
        """Handle a human clicking a preview in Human-Agent mode.

        Sends the selection back to the agent thread via agent_queue.
        """
        if self._human_agent_pending_step is None or self._agent is None:
            return

        # Deactivate the preview
        msg.is_selectable = False
        self._active_preview_msg = None

        # Send selection to agent
        self._agent.agent_queue.put({"action": "select", "selected_index": index})

        self._set_status(
            f"Selected: {pitem.label}",
            MessageContentType.STATUS,
        )

        # Clear Human-Agent step state
        self._human_agent_pending_step = None
        self._human_agent_candidates = None
        self._human_agent_effect_variants = None
        self._agent_state = AgentState.EXECUTING


    # ------------------------------------------------------------------
    # Main render (horizontal layout)
    # ------------------------------------------------------------------

    def _render_plan_bullets(self):
        """Render the concise bullet-point plan at the top."""
        if not self._plan_bullets:
            return
        imgui.push_style_color(
            imgui.COLOR_TEXT,
            _SYSTEM_TEXT_COLOR[0],
            _SYSTEM_TEXT_COLOR[1],
            _SYSTEM_TEXT_COLOR[2],
            _SYSTEM_TEXT_COLOR[3],
        )
        for bullet in self._plan_bullets:
            imgui.text(bullet)
        imgui.pop_style_color()
        imgui.spacing()

    def _render_step_title(self):
        """Render the current step title (e.g. 'Step 2/4: Adding Gaussian Blur')."""
        if self._current_step_index < 0 or not self._current_effect_name:
            return
        if self._current_operation == "remove":
            verb = "Removing"
        elif self._current_operation == "modify":
            verb = "Adjusting"
        else:
            verb = "Adding"
        title = (
            f"Step {self._current_step_index + 1}/{self._plan_step_count}: "
            f"{verb} {self._current_effect_name}"
        )
        imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0, 1.0)
        imgui.text(title)
        imgui.pop_style_color()
        imgui.spacing()

    def _render_horizontal_previews(self, effects_processor):
        """Render the 5 preview options horizontally above the input box."""
        if self._active_preview_msg is None or not self._active_preview_msg.is_selectable:
            return
        msg = self._active_preview_msg

        # Ensure textures are generated
        if msg.preview_source == "lut" and self._active_lut_results:
            self._ensure_preview_textures(
                effects_processor, self._active_lut_results, self._lut_preview_generator
            )
        elif msg.preview_source == "effect" and self._active_effect_variants:
            self._ensure_preview_textures(
                effects_processor,
                self._active_effect_variants,
                self._effect_preview_generator,
            )

        avail_width = imgui.get_content_region_available()[0]
        num_items = len(msg.preview_items)
        if num_items == 0:
            return

        spacing = 8.0
        # Divide available width evenly among items
        item_width = (avail_width - spacing * (num_items - 1)) / num_items
        if item_width < 40:
            item_width = 40
        button_width = item_width
        button_height = button_width / max(self.preview_base_aspect_ratio, 0.1)
        # Cap height to avoid oversized previews
        max_height = 200.0
        if button_height > max_height:
            button_height = max_height
            button_width = button_height * self.preview_base_aspect_ratio

        # Hover tooltip size
        hover_max_size = 384
        if self.preview_base_aspect_ratio > 1.0:
            hover_width = hover_max_size
            hover_height = hover_max_size / self.preview_base_aspect_ratio
        else:
            hover_height = hover_max_size
            hover_width = hover_max_size * self.preview_base_aspect_ratio

        selected_item = None
        selected_index = None

        for i, pitem in enumerate(msg.preview_items):
            texture_name = f"preview_{i}"
            texture_id = TextureManager.get_texture_id(texture_name)
            if texture_id is None:
                continue

            if i > 0:
                imgui.same_line(spacing=spacing)

            imgui.begin_group()

            # Short label above image
            label = pitem.label
            max_label_width = button_width
            text_width = imgui.calc_text_size(label)[0]
            if text_width > max_label_width:
                ellipsis = "..."
                ellipsis_w = imgui.calc_text_size(ellipsis)[0]
                budget = max_label_width - ellipsis_w
                while label and imgui.calc_text_size(label)[0] > budget:
                    label = label[:-1]
                label = label.rstrip() + ellipsis
            imgui.push_style_color(
                imgui.COLOR_TEXT,
                _SYSTEM_TEXT_COLOR[0],
                _SYSTEM_TEXT_COLOR[1],
                _SYSTEM_TEXT_COLOR[2],
                _SYSTEM_TEXT_COLOR[3],
            )
            imgui.text(label)
            imgui.pop_style_color()

            if imgui.image_button(
                texture_id, button_width, button_height, uv0=(0, 0), uv1=(1, 1)
            ):
                selected_item = pitem
                selected_index = i

            # Enlarged preview on hover
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.image(texture_id, hover_width, hover_height)
                imgui.end_tooltip()

            imgui.end_group()

        imgui.spacing()

        # Handle selection
        if selected_item is not None:
            self._handle_preview_selection(msg, selected_item, selected_index)

    _SPINNER_CHARS = "|/-\\"

    def _render_status_line(self):
        """Render a brief status/error summary line with spinner when agent is active."""
        if not self._status_text:
            return
        if self._status_type == MessageContentType.ERROR:
            color = _ERROR_COLOR
        elif self._status_type == MessageContentType.STATUS:
            color = _STATUS_COLOR
        else:
            color = _SYSTEM_TEXT_COLOR

        # Show animated spinner when agent is busy
        spinner = ""
        if self._agent_state in (AgentState.PLANNING, AgentState.EXECUTING):
            idx = int(time.time() * 8) % len(self._SPINNER_CHARS)
            spinner = self._SPINNER_CHARS[idx] + " "

        imgui.push_style_color(imgui.COLOR_TEXT, color[0], color[1], color[2], color[3])
        imgui.text(f"{spinner}{self._status_text}")
        imgui.pop_style_color()

    def _render_reference_image_indicator(self):
        """Show a small thumbnail with filename and remove button when a reference image is attached."""
        if self._reference_image_tensor is None or not self._reference_texture_registered:
            return

        texture_id = None
        try:
            texture_id = TextureManager.get_texture_id(self._REFERENCE_TEXTURE_NAME)
        except (ValueError, AttributeError):
            return
        if texture_id is None:
            return

        # Layout: [thumbnail] filename  [x]
        thumb_h = 28.0
        C, H, W = self._reference_image_tensor.shape
        aspect = W / H
        thumb_w = thumb_h * aspect

        imgui.begin_group()
        imgui.image(texture_id, thumb_w, thumb_h)
        imgui.same_line()

        # Show filename in muted color
        filename = os.path.basename(self._reference_image_path or "reference")
        imgui.push_style_color(
            imgui.COLOR_TEXT,
            _MODE_LABEL_COLOR[0], _MODE_LABEL_COLOR[1],
            _MODE_LABEL_COLOR[2], _MODE_LABEL_COLOR[3],
        )
        # Vertically center text with thumbnail
        cursor_y = imgui.get_cursor_pos_y()
        text_h = imgui.get_text_line_height()
        imgui.set_cursor_pos_y(cursor_y + (thumb_h - text_h) * 0.5)
        imgui.text(filename)
        imgui.pop_style_color()

        imgui.same_line()
        # Small remove button
        imgui.set_cursor_pos_y(cursor_y + (thumb_h - text_h) * 0.5)
        imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.4, 0.4, 1.0)
        if imgui.small_button("x##remove_ref"):
            self._remove_reference_image()
        imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Remove reference image")

        imgui.end_group()

        # Enlarged preview on hover over thumbnail
        # (Re-check hover on the group area is tricky; hover on the image itself)

    def _render_options_row(self):
        """Render the settings controls below the input box."""
        imgui.push_style_color(
            imgui.COLOR_TEXT,
            _MODE_LABEL_COLOR[0],
            _MODE_LABEL_COLOR[1],
            _MODE_LABEL_COLOR[2],
            _MODE_LABEL_COLOR[3],
        )

        # Model selector
        imgui.text("Model:")
        imgui.pop_style_color()
        imgui.same_line()
        _MODEL_OPTIONS = ["Auto", "Flash", "Pro"]
        _MODEL_KEYS = ["auto", "flash", "pro"]
        current_model_idx = (
            _MODEL_KEYS.index(self.model_choice)
            if self.model_choice in _MODEL_KEYS
            else 0
        )
        imgui.push_item_width(80)
        model_changed, new_model_idx = imgui.combo(
            "##model_choice", current_model_idx, _MODEL_OPTIONS
        )
        imgui.pop_item_width()
        if model_changed:
            self.model_choice = _MODEL_KEYS[new_model_idx]
            self._save_settings()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Auto: Pro for plans with >2 steps, Flash otherwise.\n"
                "Flash: always the Flash model.\n"
                "Pro: always the Pro model."
            )

        imgui.same_line()
        imgui.push_style_color(
            imgui.COLOR_TEXT,
            _MODE_LABEL_COLOR[0],
            _MODE_LABEL_COLOR[1],
            _MODE_LABEL_COLOR[2],
            _MODE_LABEL_COLOR[3],
        )
        imgui.text("Steps:")
        imgui.pop_style_color()
        imgui.same_line()
        imgui.push_item_width(60)
        step_options = [str(i) for i in range(1, 7)]
        current_steps_idx = max(0, min(self.agent_max_steps - 1, 5))
        steps_changed, new_steps_idx = imgui.combo(
            "##agent_max_steps", current_steps_idx, step_options
        )
        imgui.pop_item_width()
        if steps_changed:
            self.agent_max_steps = new_steps_idx + 1
            self._save_settings()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Maximum number of editing steps per plan.")

        imgui.same_line()
        auto_changed, auto_val = imgui.checkbox("Auto", self.agent_auto_select)
        if auto_changed:
            self.agent_auto_select = auto_val
            self._save_settings()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Auto mode: agent selects from the 5 options via vision,\n"
                "refines parameters, and always does at least 1 review."
            )

        imgui.same_line()
        if imgui.button("Clear"):
            self.chat_messages.clear()
            self._clear_active_previews()
            self._plan_bullets.clear()
            self._current_step_index = -1
            self._current_effect_name = ""
            self._current_operation = ""
            self._status_text = ""
            self.save_chat_history()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Clear chat history and reset state.")

        # Stop button when agent is running
        if self._agent_running:
            imgui.same_line()
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.6, 0.15, 0.15, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.75, 0.25, 0.25, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.5, 0.10, 0.10, 1.0)
            if imgui.button("Stop"):
                if self._agent is not None:
                    self._agent.cancel()
                    if self._agent_state == AgentState.AWAITING_HUMAN_SELECTION:
                        self._agent.agent_queue.put({"action": "skip"})
                        self._clear_active_previews()
                        self._human_agent_pending_step = None
                        self._human_agent_candidates = None
                        self._human_agent_effect_variants = None
                self._agent_running = False
                self._agent_state = AgentState.AWAITING_USER
                self._set_status("Agent stopped.", MessageContentType.STATUS)
            imgui.pop_style_color(3)

        # Reference image upload button
        imgui.same_line()
        if self._reference_image_tensor is not None:
            # Show as active/highlighted when a reference image is loaded
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.4, 0.6, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.3, 0.5, 0.7, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.15, 0.35, 0.55, 1.0)
            if imgui.button("Ref"):
                self._load_reference_image()
            imgui.pop_style_color(3)
        else:
            if imgui.button("Ref"):
                self._load_reference_image()
        if imgui.is_item_hovered():
            if self._reference_image_path:
                imgui.set_tooltip(
                    f"Reference: {os.path.basename(self._reference_image_path)}\n"
                    "Click to replace. Remove via the 'x' above."
                )
            else:
                imgui.set_tooltip("Upload a reference image for the agent to match.")

    def _set_status(self, text: str, content_type: MessageContentType = MessageContentType.STATUS):
        """Set the brief status line and also append to chat history for persistence."""
        self._status_text = text
        self._status_type = content_type
        self.add_system_message(text, content_type)

    def render(self, effects_processor=None):
        """Render the horizontal chat interface."""
        # Deferred texture cleanup (must happen before any draw commands)
        if self._reference_texture_pending_delete:
            try:
                TextureManager.delete_texture(self._REFERENCE_TEXTURE_NAME)
            except (ValueError, Exception):
                pass
            self._reference_texture_registered = False
            self._reference_texture_pending_delete = False

        imgui.begin("Chat")

        # 1. Plan bullets (concise, shown after planning)
        self._render_plan_bullets()

        # 2. Step title (e.g. "Step 2/4: Adding Gaussian Blur")
        self._render_step_title()

        # 3. Status summary line (errors, undo, etc.)
        self._render_status_line()

        # 4. Five preview options horizontally
        self._render_horizontal_previews(effects_processor)

        # 5. Reference image indicator (above input when attached)
        self._render_reference_image_indicator()

        # 6. Input box (full width, horizontal)
        imgui.push_item_width(-1)
        enter_pressed, new_text = imgui.input_text(
            "##chat_input",
            self.input_text,
            flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
            | imgui.INPUT_TEXT_CALLBACK_HISTORY,
            callback=self._history_callback,
        )
        imgui.pop_item_width()

        # 7. Options below the input
        self._render_options_row()

        self.input_text = new_text

        if enter_pressed:
            self.history_index = -1
            query = self.input_text.strip()
            self.input_text = ""
            # Clear status on new input
            self._status_text = ""

            if not query:
                self._set_status("Please enter a search query", MessageContentType.ERROR)
            elif self._agent_state == AgentState.AWAITING_HUMAN_SELECTION:
                # User typed feedback while previews are shown
                self._add_to_history(query)
                self.add_user_message(query)
                if self._agent is not None:
                    self._agent.agent_queue.put(
                        {"action": "feedback", "text": query}
                    )
                self._clear_active_previews()
                self._human_agent_pending_step = None
                self._human_agent_candidates = None
                self._human_agent_effect_variants = None
                self._agent_state = AgentState.EXECUTING
                self._set_status("Regenerating options from feedback...", MessageContentType.STATUS)
            else:
                self._add_to_history(query)
                if not self._try_shortcut(query, effects_processor):
                    self.add_user_message(query)
                    self._start_human_agent_session(query, effects_processor)

        # Poll T2S and agent status each frame
        self._poll_t2s_status()
        self._poll_agent_status()

        imgui.end()

    # ------------------------------------------------------------------
    # Stack reconstruction helper
    # ------------------------------------------------------------------

    def reconstruction_required(self, reason: str = ""):
        """Mark that effects stack reconstruction is required and capture state for undo."""
        self.effects_stack.mark_reconstruction_required()
        self.effects_stack.mark_values_changed()
        self.effects_stack.capture_state(reason)
