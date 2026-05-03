"""
Gemini-powered action classification for search queries.

Uses Gemini to interpret natural language search queries and classify them
into editor actions (add, remove, undo, redo, clear, t2s) instead of
relying on hardcoded string matching. When the action is "add", Gemini also
generates context-aware preset suggestions based on the user's intent.
"""

import json
import os
from typing import Dict, Any, List, Optional

import torch
from google.genai import types

from src.utils.gemini import get_gemini_client


# Parameter types that are not suitable for preset generation
_EXCLUDED_PARAM_TYPES = {"file", "button", "choice", "bool", "text"}

# Queries that can be resolved locally without a Gemini call
_LOCAL_COMMANDS = {
    "undo": {"action": "undo", "effect_name": None, "prompt": None, "presets": None},
    "redo": {"action": "redo", "effect_name": None, "prompt": None, "presets": None},
    "clear": {
        "action": "clear",
        "effect_name": None,
        "prompt": None,
        "presets": None,
    },
}


class GeminiActionMatcher:
    """Uses Gemini to classify search queries into editor actions."""

    VALID_ACTIONS = {"add", "remove", "undo", "redo", "clear", "t2s", "unknown"}

    def __init__(self, effect_registry: dict):
        self.client = get_gemini_client()
        self.model = "gemini-3.1-flash-lite-preview"
        self.effect_registry = effect_registry
        self._load_system_prompt()
        self._effects_list = self._build_effects_list()
        self._params_catalog = self._build_effect_params_catalog()
        self._generation_config = types.GenerateContentConfig(
            response_mime_type="application/json",
        )

    def _load_system_prompt(self) -> None:
        """Load the system prompt template from the system_prompts directory."""
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "system_prompts",
            "action_classification.txt",
        )
        with open(prompt_path, "r") as f:
            self.system_prompt_template = f.read()

    def _build_effects_list(self) -> str:
        """Build a formatted list of available effect keys for the prompt."""
        return "\n".join(f"- {key}" for key in sorted(self.effect_registry.keys()))

    def _build_effect_params_catalog(self) -> str:
        """Build a compact parameter catalog for all preset-eligible effects.

        Iterates over the effect registry, deduplicates by class name, and
        formats parameter definitions (name, type, range, default) for each
        effect that has numeric parameters suitable for preset generation.
        """
        seen_classes = {}

        for key in sorted(self.effect_registry.keys()):
            try:
                instance = self.effect_registry[key]()
                class_name = instance.__class__.__name__
                if class_name in seen_classes:
                    # Add this key as an alias
                    seen_classes[class_name]["keys"].append(key)
                    continue
                seen_classes[class_name] = {
                    "keys": [key],
                    "params": instance.get_params(),
                }
            except Exception:
                pass

        lines = []
        for class_name, info in sorted(
            seen_classes.items(), key=lambda x: x[1]["keys"][0]
        ):
            eligible = [
                p for p in info["params"] if p.get("type") not in _EXCLUDED_PARAM_TYPES
            ]
            if not eligible:
                continue

            key_str = ", ".join(info["keys"])
            lines.append(f"Effect keys ({key_str}):")
            for p in eligible:
                ptype = p["type"]
                default = p.get("default")
                pmin = p.get("min")
                pmax = p.get("max")

                # Convert tensors to lists for display
                if hasattr(default, "tolist"):
                    default = default.tolist()
                if hasattr(pmin, "tolist"):
                    pmin = pmin.tolist()
                if hasattr(pmax, "tolist"):
                    pmax = pmax.tolist()

                parts = [f"  - {p['name']} ({ptype})"]
                if pmin is not None and pmax is not None:
                    parts.append(f"range: {pmin} to {pmax}")
                if default is not None:
                    parts.append(f"default: {default}")
                lines.append(", ".join(parts))

        return "\n".join(lines)

    def _try_local_match(self, query: str) -> Optional[Dict[str, Any]]:
        """Try to resolve a query locally without calling Gemini.

        Handles exact effect name matches (fast path with hardcoded presets),
        simple commands (undo/redo/clear), and t2s: prefixes. Natural
        language queries fall through to Gemini for context-aware presets.

        Returns:
            Action result dict if locally resolved, None otherwise.
        """
        lower = query.lower().strip()

        # Exact command matches
        if lower in _LOCAL_COMMANDS:
            return _LOCAL_COMMANDS[lower]

        # Exact effect name match -> add with hardcoded preset fallback
        if lower in self.effect_registry:
            return {
                "action": "add",
                "effect_name": lower,
                "prompt": None,
                "presets": None,
            }

        # T2S prefix
        for prefix in ("t2s:", "text-to-shader:"):
            if lower.startswith(prefix):
                prompt = query[len(prefix) :].strip()
                return {
                    "action": "t2s",
                    "effect_name": None,
                    "prompt": prompt or None,
                    "presets": None,
                }

        # Remove command: "remove <effect_name>" or just "remove"
        if lower.startswith("remove"):
            rest = lower[6:].strip()
            effect_name = rest if rest in self.effect_registry else (rest or None)
            return {
                "action": "remove",
                "effect_name": effect_name,
                "prompt": None,
                "presets": None,
            }

        return None

    def _convert_preset_params(
        self, effect_name: str, presets: List[Dict]
    ) -> List[Dict]:
        """Convert raw JSON preset params to proper Python/Torch types.

        Converts [r, g, b] arrays to torch.Tensor for vec3/vec4 params,
        casts float/int values, and silently drops unknown parameter names.
        """
        try:
            instance = self.effect_registry[effect_name]()
            param_defs = {p["name"]: p for p in instance.get_params()}
        except Exception:
            return presets

        converted = []
        for preset in presets:
            if not isinstance(preset, dict):
                continue
            label = preset.get("label", "Preset")
            raw_params = preset.get("params", {})
            if not isinstance(raw_params, dict):
                continue

            new_params = {}
            for pname, pvalue in raw_params.items():
                if pname not in param_defs:
                    continue
                ptype = param_defs[pname]["type"]
                try:
                    if ptype in ("vec3", "vec4") and isinstance(pvalue, list):
                        new_params[pname] = torch.tensor(pvalue, dtype=torch.float32)
                    elif ptype == "int":
                        new_params[pname] = int(pvalue)
                    elif ptype == "float":
                        new_params[pname] = float(pvalue)
                    else:
                        new_params[pname] = pvalue
                except (ValueError, TypeError):
                    continue

            converted.append({"label": str(label), "params": new_params})
        return converted

    def _validate_preset_params(
        self, effect_name: str, presets: List[Dict]
    ) -> List[Dict]:
        """Clamp preset parameter values to their defined min/max ranges."""
        try:
            instance = self.effect_registry[effect_name]()
            param_defs = {p["name"]: p for p in instance.get_params()}
        except Exception:
            return presets

        for preset in presets:
            for pname, pvalue in preset.get("params", {}).items():
                if pname not in param_defs:
                    continue
                pdef = param_defs[pname]
                ptype = pdef["type"]
                pmin = pdef.get("min")
                pmax = pdef.get("max")
                if pmin is None or pmax is None:
                    continue

                if ptype == "float":
                    preset["params"][pname] = max(
                        float(pmin), min(float(pmax), float(pvalue))
                    )
                elif ptype == "int":
                    preset["params"][pname] = max(
                        int(pmin), min(int(pmax), int(pvalue))
                    )
                elif ptype in ("vec3", "vec4") and isinstance(pvalue, torch.Tensor):
                    tmin = (
                        pmin
                        if isinstance(pmin, torch.Tensor)
                        else torch.tensor(pmin, dtype=torch.float32)
                    )
                    tmax = (
                        pmax
                        if isinstance(pmax, torch.Tensor)
                        else torch.tensor(pmax, dtype=torch.float32)
                    )
                    preset["params"][pname] = torch.clamp(pvalue, min=tmin, max=tmax)

        return presets

    def match_action(
        self, query: str, use_gemini: bool = True, model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify a search query into an editor action.

        First tries a fast local match for exact effect names and simple
        commands (undo, redo, clear, t2s:, remove). Falls back to Gemini for
        natural language queries, which also generates context-aware
        preset suggestions.

        Args:
            query: The user's search query string.
            use_gemini: If False, skip the Gemini Flash 2.5 LLM call and
                return an "add" action with no effect_name (triggering LUT
                search fallback) when local matching fails.
            model: Gemini model ID to use (e.g. "gemini-2.5-flash"). If None,
                falls back to self.model ("gemini-2.5-flash" by default).

        Returns:
            Dictionary with keys:
                - action: One of "add", "remove", "undo", "redo", "clear", "t2s", "unknown"
                - effect_name: Matched effect key (str) or None
                - prompt: T2S prompt (str) or None
                - presets: List of {"label": str, "params": dict} or None
        """
        # Fast path: try to resolve locally without an API call
        local_result = self._try_local_match(query)
        if local_result is not None:
            return local_result

        # Offline mode: skip Gemini LLM, default to LUT search
        if not use_gemini:
            return {
                "action": "add",
                "effect_name": None,
                "prompt": None,
                "presets": None,
            }

        # Slow path: call Gemini for natural language classification
        full_prompt = self.system_prompt_template.format(
            effects_list=self._effects_list,
            user_query=query,
            params_catalog=self._params_catalog,
        )

        try:
            response = self.client.models.generate_content(
                model=model or self.model,
                contents=full_prompt,
                config=self._generation_config,
            )

            result = json.loads(
                "".join(
                    p.text
                    for p in response.candidates[0].content.parts
                    if hasattr(p, "text") and p.text
                )
            )

            # Validate action
            action = result.get("action", "unknown")
            if action not in self.VALID_ACTIONS:
                action = "unknown"

            # Normalize effect_name
            effect_name = result.get("effect_name")
            if effect_name is not None:
                effect_name = effect_name.lower().strip()
                if effect_name == "null" or effect_name == "":
                    effect_name = None

            # Normalize prompt
            prompt = result.get("prompt")
            if prompt is not None:
                prompt = str(prompt).strip()
                if prompt == "null" or prompt == "":
                    prompt = None

            # Extract and validate presets
            presets = None
            if action == "add" and effect_name is not None:
                raw_presets = result.get("presets")
                if isinstance(raw_presets, list) and len(raw_presets) > 0:
                    presets = self._convert_preset_params(effect_name, raw_presets)
                    presets = self._validate_preset_params(effect_name, presets)
                    # Drop empty presets (no valid params after conversion)
                    presets = [p for p in presets if len(p.get("params", {})) > 0]
                    if not presets:
                        print(f"Gemini presets dropped: raw={raw_presets}")
                        presets = None

            return {
                "action": action,
                "effect_name": effect_name,
                "prompt": prompt,
                "presets": presets,
            }

        except Exception as e:
            msg = str(e)
            if "404" in msg or "not found" in msg.lower():
                try:
                    available = [m.name for m in self.client.models.list()]
                    models_str = (
                        "\n  ".join(available) if available else "(none returned)"
                    )
                    msg = (
                        f"Model '{model or self.model}' not found (404).\n"
                        f"Available models:\n  {models_str}"
                    )
                except Exception:
                    msg = f"Model '{model or self.model}' not found (404). Could not list available models."
            print(f"LLM action classification failed: {msg}")
            return {
                "action": "unknown",
                "effect_name": None,
                "prompt": None,
                "presets": None,
            }
