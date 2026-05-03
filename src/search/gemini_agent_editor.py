"""
Autonomous multi-step image editor powered by Gemini vision.

GeminiAgentEditor plans a sequence of editing steps from a high-level user
prompt, generates low-resolution preset previews for each step, uses Gemini
vision to pick the best option, and posts apply commands back to the UI thread
via a queue.

Threading model
───────────────
All Gemini API calls run in a background daemon thread.  The main (ImGui)
thread polls ui_queue each frame and applies effects to the EffectStack.
The background thread blocks on agent_queue when it needs the UI to provide
preview images or confirmation that an effect was applied.

Queue protocol
──────────────
ui_queue    (agent → UI) : dicts with a "type" key
  "status"       text: str
  "error"        text: str
  "plan_ready"   goal_summary: str, step_count: int
  "step_status"  step_index: int, effect_name: str, operation: str, rationale: str
  "apply_effect" operation: str, effect_name: str, params: dict,
                 stack_index: int | None
  "done"         summary: str

agent_queue (UI → agent) : None (confirmation that apply_effect was processed)
"""

import copy
import json
import os
import queue
import re
import threading
from enum import Enum, auto
from typing import Any

import torch
import torch.nn.functional as F
from google.genai import types

from src.search.effect_presets import EffectPresetRegistry
from src.search.nearest_lut import NearestLutSearch
from src.utils.conversion import encode_image_as_jpeg, tensor_to_pil
from src.utils.gemini import get_gemini_client


# ---------------------------------------------------------------------------
# Prompt substitution helper
# ---------------------------------------------------------------------------


def _fmt_stack_value(v) -> str:
    """Format a single effect parameter value for the stack description prompt.

    Scalars are shown precisely; short flat lists (e.g. vec3) are shown as
    [r, g, b]; large or nested lists (e.g. a full LUT tensor) are replaced
    with a compact shape summary so they don't bloat the prompt or crash.
    """
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, list):
        if not v:
            return "[]"
        # Nested list (e.g. 3-D LUT stored as list-of-list-of-list)
        if isinstance(v[0], list):
            return f"<tensor {len(v)}x...>"
        # Long flat list (shouldn't normally occur, but guard anyway)
        if len(v) > 8:
            return f"<list len={len(v)}>"
        # Short flat list — format each element safely
        return "[" + ", ".join(_fmt_stack_value(x) for x in v) + "]"
    return str(v)


def _safe_format(template: str, **kwargs: str) -> str:
    """Substitute named placeholders in a prompt template without using
    Python's str.format(), which breaks when substituted values contain
    literal curly braces (e.g. JSON snippets, vec3 defaults, GLSL code).

    Replacements are done via plain str.replace() for each key, then any
    doubled braces ({{/}}) left in the template for escaping are collapsed
    back to single braces.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", value)
    result = result.replace("{{", "{").replace("}}", "}")
    return result


# ---------------------------------------------------------------------------
# State enum (informational; the main thread tracks this externally)
# ---------------------------------------------------------------------------


class AgentState(Enum):
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()
    AWAITING_USER = auto()
    AWAITING_HUMAN_SELECTION = auto()  # Human-Agent mode: waiting for human to pick
    ERROR = auto()


# ---------------------------------------------------------------------------
# Low-resolution JPEG helper
# ---------------------------------------------------------------------------


def tensor_to_vision_jpeg(tensor: torch.Tensor, max_long_edge: int = 256) -> bytes:
    """Resize a CHW float32 [0, 1] tensor so its long edge ≤ max_long_edge,
    then encode as JPEG bytes suitable for Gemini vision input."""
    C, H, W = tensor.shape
    long_edge = max(H, W)
    if long_edge > max_long_edge:
        scale = max_long_edge / long_edge
        new_h = max(1, int(H * scale))
        new_w = max(1, int(W * scale))
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    tensor = tensor.clamp(0.0, 1.0)
    pil_img = tensor_to_pil(tensor)
    return encode_image_as_jpeg(pil_img)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class GeminiAgentEditor:
    """Autonomous multi-step image editor that uses Gemini vision to select
    the best preset variant at each editing step."""

    _PROMPTS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "system_prompts",
    )

    # Effects the agent is not allowed to use
    _EXCLUDED_EFFECTS: frozenset[str] = frozenset({"searchlut"})

    def __init__(
        self,
        effect_registry: dict,
        preset_registry: EffectPresetRegistry,
        params_catalog: str,
        effects_list: str,
    ):
        self.client = get_gemini_client()
        self.model = "gemini-3.1-flash-lite-preview"
        # Full registry (used by human-agent mode which supports all effects)
        self._full_effect_registry = dict(effect_registry)
        # Filter out excluded effects so they never appear in agentic prompts
        self.effect_registry = {
            k: v for k, v in effect_registry.items() if k not in self._EXCLUDED_EFFECTS
        }
        self.preset_registry = preset_registry
        self.params_catalog = self._augment_params_catalog(
            self._filter_params_catalog(params_catalog)
        )
        self.effects_list = self._filter_effects_list(effects_list)
        # Unfiltered catalogs for human-agent mode
        self._full_params_catalog = self._augment_params_catalog(
            params_catalog, registry=self._full_effect_registry
        )
        self._full_effects_list = effects_list

        # Queues connecting agent thread ↔ main thread
        self.ui_queue: queue.Queue = queue.Queue()
        self.agent_queue: queue.Queue = queue.Queue()

        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._conversation_history: list[str] = []

        self._planning_prompt_template = self._load_prompt("agentic_planning.txt")
        self._selection_prompt_template = self._load_prompt(
            "agentic_preview_selection.txt"
        )
        self._review_prompt_template = self._load_prompt("agentic_review.txt")
        self._refinement_prompt_template = self._load_prompt(
            "agentic_param_refinement.txt"
        )
        self._human_agent_planning_template = self._load_prompt(
            "human_agent_planning.txt"
        )
        self._human_agent_review_candidates_template = self._load_prompt(
            "human_agent_review_candidates.txt"
        )
        self._human_agent_feedback_template = self._load_prompt(
            "human_agent_feedback.txt"
        )
        self._human_agent_review_template = self._load_prompt(
            "human_agent_review.txt"
        )

    # ------------------------------------------------------------------
    # Public API (called from main thread)
    # ------------------------------------------------------------------

    def start(
        self,
        user_prompt: str,
        vision_thumbnail: torch.Tensor,
        stack_snapshot: list[dict],
        max_reviews: int = 0,
        enable_refinement: bool = True,
        max_steps: int = 4,
        model: str | None = None,
        human_selection: bool = False,
        auto_select: bool = False,
        reference_image: torch.Tensor | None = None,
    ) -> None:
        """Launch (or restart) the background editing thread.

        If *human_selection* is True, the agent operates in Human-Agent mode:
        it plans edits and builds 5 candidate presets per step, reviews them
        for coherence, then sends them to the UI for the human to choose
        (instead of selecting automatically via Gemini vision).

        If *reference_image* is provided (CHW float32 [0,1] tensor), it will
        be sent alongside the current image so the agent can match the
        reference's look and feel.
        """
        self._cancel_event.clear()
        # Drain any leftover messages from a previous run
        while not self.ui_queue.empty():
            try:
                self.ui_queue.get_nowait()
            except queue.Empty:
                break
        while not self.agent_queue.empty():
            try:
                self.agent_queue.get_nowait()
            except queue.Empty:
                break

        ref_clone = reference_image.clone() if reference_image is not None else None
        target = self._run_human_agent if human_selection else self._run
        args = [
            user_prompt,
            vision_thumbnail.clone(),
            stack_snapshot,
            max_reviews,
            enable_refinement,
            max_steps,
            model,
        ]
        if human_selection:
            args.append(auto_select)
        args.append(ref_clone)
        self._thread = threading.Thread(
            target=target,
            args=tuple(args),
            daemon=True,
        )
        self._thread.start()

    def confirm_applied(self) -> None:
        """Called by the main thread after it has applied an effect to the stack."""
        self.agent_queue.put(None)

    def cancel(self) -> None:
        """Signal the background thread to stop at the next checkpoint."""
        self._cancel_event.set()

    # ------------------------------------------------------------------
    # Background thread entry point
    # ------------------------------------------------------------------

    def _run(
        self,
        user_prompt: str,
        vision_thumbnail: torch.Tensor,
        stack_snapshot: list[dict],
        max_reviews: int = 0,
        enable_refinement: bool = True,
        max_steps: int = 4,
        model: str | None = None,
        reference_image: torch.Tensor | None = None,
    ) -> None:
        if model:
            self.model = model
        try:
            # ---- PHASE 1: PLANNING ----
            result = self._execute_planning_phase(
                user_prompt, stack_snapshot, max_steps,
                self._planning_prompt_template,
                self.effects_list, self.params_catalog,
                reference_image=reference_image,
            )
            if result is None:
                return
            steps, goal_summary = result

            if not steps:
                self.ui_queue.put(
                    {"type": "done", "summary": "No edits needed for this request."}
                )
                return

            # ---- PHASE 2: EXECUTE STEPS ----
            running_thumbnail = vision_thumbnail.clone()
            # Track current stack state so the review phase can reference stack_index values
            current_stack: list[dict] = list(stack_snapshot)
            applied_summary: list[str] = []
            applied_steps: list[tuple[str, str]] = []  # (effect_name, chosen_label)

            for step in steps:
                if self._cancel_event.is_set():
                    self.ui_queue.put({"type": "status", "text": "Agent cancelled."})
                    return

                success, running_thumbnail, chosen_label, reasoning, chosen_params = (
                    self._execute_step(step, running_thumbnail, goal_summary)
                )
                if success:
                    op = step.get("operation", "add")
                    if op != "remove":
                        refine_index = (
                            len(current_stack)
                            if op == "add"
                            else step.get("stack_index")
                        )
                        if enable_refinement:
                            running_thumbnail, chosen_params = self._refine_params(
                                step,
                                chosen_label,
                                chosen_params,
                                running_thumbnail,
                                refine_index,
                                goal_summary,
                            )
                    self._update_stack_snapshot(current_stack, step, chosen_params)
                    applied_summary.append(
                        f"- {op} {step.get('effect_name', '')} "
                        f'as "{chosen_label}": {reasoning}'
                    )
                    applied_steps.append((step.get("effect_name", ""), chosen_label))

            # ---- PHASE 3: ITERATIVE REVIEW ----
            for _cycle in range(max_reviews):
                if self._cancel_event.is_set():
                    break

                self.ui_queue.put({"type": "status", "text": "Reviewing result..."})

                review = self._call_review(
                    user_prompt,
                    goal_summary,
                    applied_summary,
                    running_thumbnail,
                    current_stack,
                )
                if review is None or review.get("satisfied", True):
                    break

                critique = review.get("critique", "")
                additional_steps = review.get("additional_steps", [])
                if not additional_steps:
                    break

                self.ui_queue.put(
                    {
                        "type": "review_plan",
                        "critique": critique,
                        "steps": additional_steps,
                    }
                )

                for step in additional_steps:
                    if self._cancel_event.is_set():
                        break
                    (
                        success,
                        running_thumbnail,
                        chosen_label,
                        reasoning,
                        chosen_params,
                    ) = self._execute_step(step, running_thumbnail, goal_summary)
                    if success:
                        op = step.get("operation", "add")
                        if op != "remove":
                            refine_index = (
                                len(current_stack)
                                if op == "add"
                                else step.get("stack_index")
                            )
                            if enable_refinement:
                                running_thumbnail, chosen_params = self._refine_params(
                                    step,
                                    chosen_label,
                                    chosen_params,
                                    running_thumbnail,
                                    refine_index,
                                    goal_summary,
                                )
                        self._update_stack_snapshot(current_stack, step, chosen_params)
                        applied_summary.append(
                            f"- {op} {step.get('effect_name', '')} "
                            f'as "{chosen_label}" (revision): {reasoning}'
                        )
                        applied_steps.append(
                            (step.get("effect_name", ""), chosen_label)
                        )

            # ---- PHASE 4: SUMMARY ----
            if not self._cancel_event.is_set():
                self._execute_summary_phase(user_prompt, goal_summary, applied_steps)

        except Exception as exc:
            self.ui_queue.put({"type": "error", "text": f"Agent error: {exc}"})

    # ------------------------------------------------------------------
    # Human-Agent mode: background thread entry point
    # ------------------------------------------------------------------

    def _run_human_agent(
        self,
        user_prompt: str,
        vision_thumbnail: torch.Tensor,
        stack_snapshot: list[dict],
        max_reviews: int = 0,
        enable_refinement: bool = True,
        max_steps: int = 4,
        model: str | None = None,
        auto_select: bool = False,
        reference_image: torch.Tensor | None = None,
    ) -> None:
        """Human-Agent mode: plan edits, present 5 options per step for human selection."""
        if model:
            self.model = model
        try:
            # ---- PHASE 1: PLANNING ----
            result = self._execute_planning_phase(
                user_prompt, stack_snapshot, max_steps,
                self._human_agent_planning_template,
                self._full_effects_list, self._full_params_catalog,
                reference_image=reference_image,
            )
            if result is None:
                return
            steps, goal_summary = result

            if not steps:
                self.ui_queue.put(
                    {"type": "done", "summary": "No edits needed for this request."}
                )
                return

            # ---- PHASE 2: EXECUTE STEPS (human selects at each step) ----
            running_thumbnail = vision_thumbnail.clone()
            current_stack: list[dict] = list(stack_snapshot)
            applied_steps: list[tuple[str, str]] = []

            # In auto-select mode, always do at least 1 review and enable refinement
            effective_reviews = max(1, max_reviews) if auto_select else max_reviews

            for step in steps:
                if self._cancel_event.is_set():
                    self.ui_queue.put({"type": "status", "text": "Agent cancelled."})
                    return

                success, running_thumbnail, chosen_label, chosen_params = (
                    self._execute_human_agent_step(
                        step, running_thumbnail, goal_summary,
                        auto_select=auto_select,
                    )
                )
                if success:
                    op = step.get("operation", "add")
                    if auto_select and op != "remove":
                        refine_index = (
                            len(current_stack)
                            if op == "add"
                            else step.get("stack_index")
                        )
                        running_thumbnail, chosen_params = self._refine_params(
                            step,
                            chosen_label,
                            chosen_params,
                            running_thumbnail,
                            refine_index,
                            goal_summary,
                        )
                    self._update_stack_snapshot(current_stack, step, chosen_params)
                    applied_steps.append(
                        (step.get("effect_name", ""), chosen_label)
                    )
                elif success is None:
                    # Human cancelled / stopped mid-step
                    break

            # ---- PHASE 3: ITERATIVE REVIEW ----
            if effective_reviews > 0 and applied_steps and not self._cancel_event.is_set():
                applied_summary: list[str] = []
                for effect_name_s, chosen_label_s in applied_steps:
                    applied_summary.append(
                        f"- add {effect_name_s} as \"{chosen_label_s}\""
                    )

                for _cycle in range(effective_reviews):
                    if self._cancel_event.is_set():
                        break

                    self.ui_queue.put(
                        {"type": "status", "text": "Reviewing result..."}
                    )

                    review = self._call_human_agent_review(
                        user_prompt,
                        goal_summary,
                        applied_summary,
                        running_thumbnail,
                        current_stack,
                    )
                    if review is None or review.get("satisfied", True):
                        break

                    critique = review.get("critique", "")
                    additional_steps = review.get("additional_steps", [])
                    if not additional_steps:
                        break

                    # Show the review plan to the user
                    self.ui_queue.put(
                        {
                            "type": "review_plan",
                            "critique": critique,
                            "steps": additional_steps,
                        }
                    )

                    for step in additional_steps:
                        if self._cancel_event.is_set():
                            break

                        success, running_thumbnail, chosen_label, chosen_params = (
                            self._execute_human_agent_step(
                                step, running_thumbnail, goal_summary,
                                auto_select=auto_select,
                            )
                        )
                        if success:
                            op = step.get("operation", "add")
                            if auto_select and op != "remove":
                                refine_index = (
                                    len(current_stack)
                                    if op == "add"
                                    else step.get("stack_index")
                                )
                                running_thumbnail, chosen_params = self._refine_params(
                                    step,
                                    chosen_label,
                                    chosen_params,
                                    running_thumbnail,
                                    refine_index,
                                    goal_summary,
                                )
                            self._update_stack_snapshot(
                                current_stack, step, chosen_params
                            )
                            applied_steps.append(
                                (step.get("effect_name", ""), chosen_label)
                            )
                            applied_summary.append(
                                f"- {op} "
                                f"{step.get('effect_name', '')} "
                                f'as "{chosen_label}" (revision): {critique}'
                            )
                        elif success is None:
                            break

            # ---- PHASE 4: SUMMARY ----
            if not self._cancel_event.is_set():
                self._execute_summary_phase(user_prompt, goal_summary, applied_steps)

        except Exception as exc:
            self.ui_queue.put({"type": "error", "text": f"Agent error: {exc}"})

    def _execute_human_agent_step(
        self,
        step: dict,
        running_thumbnail: torch.Tensor,
        goal_summary: str,
        auto_select: bool = False,
    ) -> tuple[bool | None, torch.Tensor, str, dict]:
        """Execute a single step in Human-Agent mode.

        Builds 5 candidates, reviews them for coherence via Gemini, then
        sends them to the UI for the human to select (or auto-selects via
        vision when *auto_select* is True).

        Returns (success, updated_thumbnail, chosen_label, chosen_params).
        success=None means the human cancelled.
        """
        effect_name = step.get("effect_name", "")
        operation = step.get("operation", "add")
        step_index = step.get("step_index", 0)
        rationale = step.get("rationale", "")

        self.ui_queue.put(
            {
                "type": "step_status",
                "step_index": step_index,
                "effect_name": effect_name,
                "operation": operation,
                "rationale": rationale,
            }
        )

        # Remove operations need no candidates — just delete the effect
        if operation == "remove":
            self.ui_queue.put(
                {
                    "type": "apply_effect",
                    "operation": "remove",
                    "effect_name": effect_name,
                    "params": {},
                    "stack_index": step.get("stack_index"),
                }
            )
            self.agent_queue.get()
            self.ui_queue.put(
                {
                    "type": "status",
                    "text": f"Removed {effect_name}",
                }
            )
            return True, running_thumbnail, "Removed", {}

        # Build candidates (aim for 5)
        registry = self._full_effect_registry
        candidates = self._build_candidates(step, registry=registry, max_candidates=5)
        if not candidates:
            self.ui_queue.put(
                {
                    "type": "status",
                    "text": f"Skipping {operation} {effect_name}: no valid candidates.",
                }
            )
            return False, running_thumbnail, "", {}

        # Review all candidates together via Gemini for coherence
        # (skip for searchlut — candidates come from semantic search, not params)
        if effect_name != "searchlut":
            candidates = self._review_candidates(
                step, candidates, goal_summary, effect_name, registry=registry
            )

        if auto_select:
            # Select via vision — no previews sent to the UI
            chosen_idx, _reasoning = self._select_via_vision(
                step, candidates, running_thumbnail, goal_summary
            )
        else:
            # Build candidate_data for UI previews
            candidate_data = []
            for label, eff in candidates:
                params = self._extract_params(eff)
                cand_entry = {
                    "label": label, "params": params, "effect_name": effect_name
                }
                if effect_name == "searchlut" and hasattr(eff, "lut_tensor"):
                    cand_entry["lut_data"] = {
                        "lut_tensor": eff.lut_tensor,
                        "domain_min": eff.domain_min,
                        "domain_max": eff.domain_max,
                    }
                candidate_data.append(cand_entry)

            # Send candidates to UI for display
            self.ui_queue.put(
                {
                    "type": "step_previews",
                    "step_index": step_index,
                    "effect_name": effect_name,
                    "operation": operation,
                    "stack_index": step.get("stack_index"),
                    "rationale": rationale,
                    "candidates": candidate_data,
                }
            )

            # Feedback loop: wait for human selection or feedback
            while True:
                response = self.agent_queue.get()
                if self._cancel_event.is_set():
                    return None, running_thumbnail, "", {}

                if response is None or response.get("action") == "skip":
                    self.ui_queue.put(
                        {"type": "status", "text": f"Skipped step {step_index + 1}."}
                    )
                    return False, running_thumbnail, "", {}

                if response.get("action") == "feedback":
                    # Human provided feedback — regenerate candidates
                    feedback_text = response.get("text", "")
                    self.ui_queue.put(
                        {
                            "type": "status",
                            "text": f"Regenerating options with feedback: \"{feedback_text[:60]}...\"",
                        }
                    )
                    new_candidates = self._regenerate_with_feedback(
                        step, candidates, goal_summary, effect_name, feedback_text,
                        registry=registry,
                    )
                    if new_candidates:
                        candidates = new_candidates
                        # Rebuild candidate_data and resend previews
                        candidate_data = []
                        for lbl, ef in candidates:
                            p = self._extract_params(ef)
                            cd = {"label": lbl, "params": p, "effect_name": effect_name}
                            if effect_name == "searchlut" and hasattr(ef, "lut_tensor"):
                                cd["lut_data"] = {
                                    "lut_tensor": ef.lut_tensor,
                                    "domain_min": ef.domain_min,
                                    "domain_max": ef.domain_max,
                                }
                            candidate_data.append(cd)
                        self.ui_queue.put(
                            {
                                "type": "step_previews",
                                "step_index": step_index,
                                "effect_name": effect_name,
                                "operation": operation,
                                "stack_index": step.get("stack_index"),
                                "rationale": rationale,
                                "candidates": candidate_data,
                            }
                        )
                    continue

                # action == "select"
                chosen_idx = response.get("selected_index", 0)
                break

        chosen_idx = max(0, min(chosen_idx, len(candidates) - 1))
        chosen_label, chosen_effect = candidates[chosen_idx]
        chosen_params = self._extract_params(chosen_effect)

        # Apply the chosen effect
        apply_msg = {
            "type": "apply_effect",
            "operation": operation,
            "effect_name": effect_name,
            "params": chosen_params,
            "stack_index": step.get("stack_index"),
        }
        # Pass pre-loaded LUT data so the window reproduces the exact match
        if effect_name == "searchlut" and hasattr(chosen_effect, "lut_tensor"):
            apply_msg["lut_data"] = {
                "lut_tensor": chosen_effect.lut_tensor,
                "domain_min": chosen_effect.domain_min,
                "domain_max": chosen_effect.domain_max,
            }
        self.ui_queue.put(apply_msg)
        # Wait for confirmation that apply_effect was processed
        self.agent_queue.get()

        new_thumbnail = self._apply_to_thumbnail(chosen_effect, running_thumbnail)
        self.ui_queue.put(
            {
                "type": "status",
                "text": f"Step {step_index + 1} applied: {chosen_label}",
            }
        )
        return True, new_thumbnail, chosen_label, chosen_params

    def _build_searchlut_candidates(
        self, step: dict
    ) -> list[tuple[str, Any]]:
        """Build searchlut candidates by performing semantic LUT search.

        Extracts the search_query from the step's candidate_param_sets (or the
        step rationale as fallback), runs NearestLutSearch.search_top_k, and
        returns 5 SearchLUT instances pre-loaded with the matched LUT data.
        """
        # Determine the search query — prefer the first candidate's search_query
        search_query = ""
        for cand in step.get("candidate_param_sets", []):
            sq = cand.get("params", {}).get("search_query", "")
            if sq:
                search_query = sq
                break
        if not search_query:
            search_query = step.get("rationale", "cinematic look")

        try:
            lut_search = NearestLutSearch()
            top_luts = lut_search.search_top_k(search_query, k=5)
        except Exception as exc:
            print(f"LUT search failed: {exc}")
            return []

        registry = self._full_effect_registry
        result = []
        for lut_data in top_luts:
            eff = registry["searchlut"]()
            eff.search_query = search_query
            eff.strength = 1.0
            # Pre-load the LUT data so the effect doesn't re-search
            eff.lut_tensor = torch.from_numpy(lut_data["lut_tensor"])
            eff.domain_min = lut_data["domain_min"]
            eff.domain_max = lut_data["domain_max"]
            eff.previous_search_query = search_query
            lut_name = lut_data.get("lut_name", "LUT")
            result.append((lut_name, eff))
        return result

    def _review_candidates(
        self,
        step: dict,
        candidates: list[tuple[str, Any]],
        goal_summary: str,
        effect_name: str,
        registry: dict | None = None,
    ) -> list[tuple[str, Any]]:
        """Ask Gemini to review all candidates together for coherence and diversity.

        If Gemini finds issues, it returns refined candidates which we rebuild.
        """
        if len(candidates) < 2:
            return candidates
        if registry is None:
            registry = self.effect_registry

        candidates_description = self._format_candidates_description(candidates)
        effect_params_str = self._describe_effect_params(effect_name)

        review_prompt = _safe_format(
            self._human_agent_review_candidates_template,
            goal_summary=goal_summary,
            step_rationale=step.get("rationale", ""),
            effect_name=effect_name,
            candidates_description=candidates_description,
            effect_params=effect_params_str,
        )

        result = self._call_gemini_json(review_prompt)
        if result is None or result.get("all_good", True):
            return candidates

        refined = result.get("refined_candidates", [])
        if not refined:
            return candidates

        new_candidates = []
        for cand in refined[:5]:
            label = cand.get("label", "Option")
            eff = self._instantiate_effect(
                effect_name, cand.get("params", {}), registry=registry
            )
            if eff is not None:
                new_candidates.append((label, eff))

        return new_candidates if new_candidates else candidates

    def _regenerate_with_feedback(
        self,
        step: dict,
        current_candidates: list[tuple[str, Any]],
        goal_summary: str,
        effect_name: str,
        feedback: str,
        registry: dict | None = None,
    ) -> list[tuple[str, Any]] | None:
        """Regenerate candidates using human feedback via Gemini."""
        if registry is None:
            registry = self.effect_registry

        previous_candidates = self._format_candidates_description(current_candidates)
        effect_params_str = self._describe_effect_params(effect_name)

        prompt = _safe_format(
            self._human_agent_feedback_template,
            goal_summary=goal_summary,
            step_rationale=step.get("rationale", ""),
            effect_name=effect_name,
            previous_candidates=previous_candidates,
            effect_params=effect_params_str,
            feedback=feedback,
        )

        result = self._call_gemini_json(prompt)
        if result is None:
            return None

        raw_candidates = result.get("candidates", [])
        if not raw_candidates:
            return None

        new_candidates = []
        for cand in raw_candidates[:5]:
            label = cand.get("label", "Option")
            eff = self._instantiate_effect(
                effect_name, cand.get("params", {}), registry=registry
            )
            if eff is not None:
                new_candidates.append((label, eff))

        return new_candidates if new_candidates else None

    # ------------------------------------------------------------------
    # Agentic mode step execution
    # ------------------------------------------------------------------

    def _execute_step(
        self,
        step: dict,
        running_thumbnail: torch.Tensor,
        goal_summary: str,
    ) -> tuple[bool, torch.Tensor, str, str, dict]:
        """Execute a single plan step via vision-based selection.

        Returns (success, updated_thumbnail, chosen_label, reasoning, chosen_params).
        On failure (no candidates) returns (False, original_thumbnail, "", "", {}).
        """
        effect_name = step.get("effect_name", "")
        operation = step.get("operation", "add")
        step_index = step.get("step_index", 0)
        rationale = step.get("rationale", "")

        self.ui_queue.put(
            {
                "type": "step_status",
                "step_index": step_index,
                "effect_name": effect_name,
                "operation": operation,
                "rationale": rationale,
            }
        )

        # Remove operations need no candidates — just delete the effect
        if operation == "remove":
            self.ui_queue.put(
                {
                    "type": "apply_effect",
                    "operation": "remove",
                    "effect_name": effect_name,
                    "params": {},
                    "stack_index": step.get("stack_index"),
                }
            )
            self.agent_queue.get()
            self.ui_queue.put(
                {
                    "type": "status",
                    "text": f"Step {step_index + 1} - Removed {effect_name}",
                }
            )
            return True, running_thumbnail, "Removed", rationale, {}

        candidates = self._build_candidates(step)
        if not candidates:
            self.ui_queue.put(
                {
                    "type": "status",
                    "text": f"Skipping {operation} {effect_name}: no valid candidates.",
                }
            )
            return False, running_thumbnail, "", "", {}

        chosen_idx, reasoning = self._select_via_vision(
            step, candidates, running_thumbnail, goal_summary
        )
        chosen_label, chosen_effect = candidates[chosen_idx]
        chosen_params = self._extract_params(chosen_effect)

        self.ui_queue.put(
            {
                "type": "apply_effect",
                "operation": operation,
                "effect_name": effect_name,
                "params": chosen_params,
                "stack_index": step.get("stack_index"),
            }
        )
        self.agent_queue.get()

        new_thumbnail = self._apply_to_thumbnail(chosen_effect, running_thumbnail)
        self.ui_queue.put(
            {
                "type": "status",
                "text": f"Step {step_index + 1} - {chosen_label}: {reasoning}",
            }
        )
        return True, new_thumbnail, chosen_label, reasoning, chosen_params

    def _update_stack_snapshot(
        self, current_stack: list[dict], step: dict, chosen_params: dict
    ) -> None:
        """Keep current_stack in sync with what the main thread applied."""
        operation = step.get("operation", "add")
        effect_name = step.get("effect_name", "")
        if operation == "add":
            current_stack.append({"effect": effect_name, "state": dict(chosen_params)})
        elif operation == "modify":
            idx = step.get("stack_index")
            if idx is not None and 0 <= idx < len(current_stack):
                current_stack[idx]["state"].update(chosen_params)
        elif operation == "remove":
            idx = step.get("stack_index")
            if idx is not None and 0 <= idx < len(current_stack):
                current_stack.pop(idx)

    def _call_review(
        self,
        user_prompt: str,
        goal_summary: str,
        applied_summary: list[str],
        thumbnail: torch.Tensor,
        current_stack: list[dict],
    ) -> dict | None:
        """Ask Gemini vision to evaluate the current result and suggest follow-up steps."""
        applied_text = "\n".join(applied_summary) if applied_summary else "None"
        stack_desc = self._describe_stack(current_stack)

        review_prompt = _safe_format(
            self._review_prompt_template,
            user_prompt=user_prompt,
            goal_summary=goal_summary,
            applied_steps=applied_text,
            stack_description=stack_desc,
            effects_list=self.effects_list,
            params_catalog=self.params_catalog,
        )

        jpeg = tensor_to_vision_jpeg(thumbnail)
        content_parts: list[Any] = [
            review_prompt,
            types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
        ]
        return self._call_gemini_json_with_parts(content_parts)

    def _call_human_agent_review(
        self,
        user_prompt: str,
        goal_summary: str,
        applied_summary: list[str],
        thumbnail: torch.Tensor,
        current_stack: list[dict],
    ) -> dict | None:
        """Ask Gemini vision to evaluate the result and suggest corrections with 5 candidates."""
        applied_text = "\n".join(applied_summary) if applied_summary else "None"
        stack_desc = self._describe_stack(current_stack)

        review_prompt = _safe_format(
            self._human_agent_review_template,
            user_prompt=user_prompt,
            goal_summary=goal_summary,
            applied_steps=applied_text,
            stack_description=stack_desc,
            effects_list=self._full_effects_list,
            params_catalog=self._full_params_catalog,
        )

        jpeg = tensor_to_vision_jpeg(thumbnail)
        content_parts: list[Any] = [
            review_prompt,
            types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
        ]
        return self._call_gemini_json_with_parts(content_parts)

    # ------------------------------------------------------------------
    # Candidate effect building
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        step: dict,
        registry: dict | None = None,
        max_candidates: int = 3,
    ) -> list[tuple[str, Any]]:
        """Build (label, ImageEffect) pairs from a plan step.

        Uses the step's candidate_param_sets if provided, otherwise falls back
        to the EffectPresetRegistry.  For searchlut steps, performs a semantic
        LUT search and returns the top matches as pre-loaded effects.
        """
        if registry is None:
            registry = self.effect_registry
        effect_name = step.get("effect_name", "")
        if effect_name not in registry:
            return []

        # Special handling: searchlut uses semantic LUT search
        if effect_name == "searchlut":
            return self._build_searchlut_candidates(step)

        raw_candidates = step.get("candidate_param_sets", [])

        if not raw_candidates:
            return self._build_candidates_from_presets(
                effect_name, max_candidates, registry
            )

        result = []
        for cand in raw_candidates[:max_candidates]:
            label = cand.get("label", "Option")
            eff = self._instantiate_effect(
                effect_name, cand.get("params", {}), registry=registry
            )
            if eff is not None:
                result.append((label, eff))
        return result

    # ------------------------------------------------------------------
    # Vision-based preset selection
    # ------------------------------------------------------------------

    def _select_via_vision(
        self,
        step: dict,
        candidates: list[tuple[str, Any]],
        base_thumbnail: torch.Tensor,
        goal_summary: str,
    ) -> tuple[int, str]:
        """Render candidate previews and ask Gemini vision to pick the best."""
        _, H, W = base_thumbnail.shape
        preview_jpegs: list[bytes] = []

        for label, eff in candidates:
            eff_copy = copy.deepcopy(eff)
            eff_copy.adjust_parameters_for_preview((H, W), (H, W))
            try:
                result = eff_copy.apply(base_thumbnail.clone())
            except Exception:
                result = base_thumbnail.clone()
            jpeg = tensor_to_vision_jpeg(result.clamp(0.0, 1.0))
            preview_jpegs.append(jpeg)

        n = len(candidates)
        option_descriptions = "\n".join(
            f"Option {i}: {label}" for i, (label, _) in enumerate(candidates)
        )
        selection_prompt = _safe_format(
            self._selection_prompt_template,
            goal_summary=goal_summary,
            effect_name=step.get("effect_name", ""),
            rationale=step.get("rationale", ""),
            n_options=str(n),
            option_descriptions=option_descriptions,
            n_options_minus_one=str(n - 1),
        )

        content_parts: list[Any] = [selection_prompt]
        for jpeg in preview_jpegs:
            content_parts.append(
                types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")
            )

        result = self._call_gemini_json_with_parts(content_parts)
        if result is None:
            return 0, "fallback to first option"

        idx = int(result.get("selected_index", 0))
        idx = max(0, min(idx, n - 1))
        reasoning = result.get("reasoning", "")
        return idx, reasoning

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_to_thumbnail(self, effect: Any, thumbnail: torch.Tensor) -> torch.Tensor:
        """Deep-copy effect, adjust for preview resolution, apply to thumbnail."""
        _, H, W = thumbnail.shape
        eff_copy = copy.deepcopy(effect)
        eff_copy.adjust_parameters_for_preview((H, W), (H, W))
        try:
            result = eff_copy.apply(thumbnail.clone())
        except Exception:
            result = thumbnail.clone()
        return result.clamp(0.0, 1.0)

    def _extract_params(self, effect: Any) -> dict:
        """Extract a dict of param_name → value from an effect instance."""
        params = {}
        for param_def in effect.get_params():
            if param_def.get("type") == "button":
                continue
            pname = param_def["name"]
            val = getattr(effect, pname)
            if isinstance(val, torch.Tensor):
                val = val.clone()
            params[pname] = val
        return params

    def _instantiate_effect(
        self,
        effect_name: str,
        raw_params: dict,
        registry: dict | None = None,
        base_params: dict | None = None,
    ) -> Any | None:
        """Create an effect instance with type-coerced parameters.

        If *base_params* is provided, those are set first (as-is), then
        *raw_params* are applied on top with type coercion.
        """
        if registry is None:
            registry = self.effect_registry
        if effect_name not in registry:
            return None
        try:
            eff = registry[effect_name]()
            param_defs = {p["name"]: p for p in eff.get_params()}
        except Exception:
            return None
        if base_params:
            for pname, pval in base_params.items():
                setattr(eff, pname, pval)
        for pname, pval in raw_params.items():
            if pname not in param_defs:
                continue
            ptype = param_defs[pname]["type"]
            try:
                if ptype in ("vec3", "vec4") and isinstance(pval, list):
                    pval = torch.tensor(pval, dtype=torch.float32)
                elif ptype == "float":
                    pval = float(pval)
                elif ptype == "int":
                    pval = int(pval)
            except (ValueError, TypeError):
                continue
            setattr(eff, pname, pval)
        return eff

    def _build_candidates_from_presets(
        self,
        effect_name: str,
        max_count: int,
        registry: dict,
    ) -> list[tuple[str, Any]]:
        """Build candidates from the preset registry, falling back to a default."""
        presets = self.preset_registry.get_presets(effect_name)[:max_count]
        if not presets:
            return [("Default", registry[effect_name]())]
        result = []
        for p in presets:
            eff = registry[effect_name]()
            for k, v in p.params.items():
                setattr(eff, k, v)
            result.append((p.label, eff))
        return result

    def _format_candidates_description(
        self, candidates: list[tuple[str, Any]]
    ) -> str:
        """Format candidates as a numbered text description for Gemini prompts."""
        lines = []
        for i, (label, eff) in enumerate(candidates):
            params = self._extract_params(eff)
            params_str = ", ".join(
                f"{k}={_fmt_stack_value(v)}" for k, v in params.items()
            )
            lines.append(f'  Option {i + 1}: "{label}" — {params_str}')
        return "\n".join(lines)

    def _planning_call_with_retry(self, call_fn) -> dict | None:
        """Call *call_fn* (which returns dict | None) up to 2 times.

        On the first failure the error message already pushed to ui_queue by
        the underlying _call_gemini_* helper is drained so the UI does not
        treat it as terminal, and a "retrying" status is shown instead.
        """
        result = call_fn()
        if result is not None or self._cancel_event.is_set():
            return result
        # First attempt failed — drain the error so the UI doesn't stop us
        self._drain_error_messages()
        self.ui_queue.put(
            {"type": "status", "text": "Planning failed, retrying..."}
        )
        return call_fn()

    def _drain_error_messages(self) -> None:
        """Remove pending 'error' messages from ui_queue, keep the rest."""
        kept: list[dict] = []
        while True:
            try:
                msg = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if msg.get("type") != "error":
                kept.append(msg)
        for msg in kept:
            self.ui_queue.put(msg)

    def _execute_planning_phase(
        self,
        user_prompt: str,
        stack_snapshot: list[dict],
        max_steps: int,
        template: str,
        effects_list: str,
        params_catalog: str,
        reference_image: torch.Tensor | None = None,
    ) -> tuple[list[dict], str] | None:
        """Run the planning phase common to both modes.

        Returns (steps, goal_summary) or None if cancelled / failed.
        If *reference_image* is provided it is sent as a JPEG alongside the
        prompt so Gemini can match the reference's look.
        """
        self.ui_queue.put(
            {"type": "status", "text": "Analysing image and planning edits..."}
        )

        stack_desc = self._describe_stack(stack_snapshot)
        conversation_history = "\n".join(self._conversation_history) or "None"

        planning_prompt = _safe_format(
            template,
            effects_list=effects_list,
            params_catalog=params_catalog,
            stack_description=stack_desc,
            user_prompt=user_prompt,
            conversation_history=conversation_history,
            max_steps=str(max_steps),
        )

        if reference_image is not None:
            ref_jpeg = tensor_to_vision_jpeg(reference_image, max_long_edge=256)
            ref_note = (
                "\n\n[A reference image is attached below. The user wants the "
                "edited photo to match the look, mood, color grading, and style "
                "of this reference image. Use it to guide your editing plan.]\n"
            )
            content_parts: list[Any] = [
                planning_prompt + ref_note,
                types.Part.from_bytes(data=ref_jpeg, mime_type="image/jpeg"),
            ]
            plan = self._planning_call_with_retry(
                lambda: self._call_gemini_json_with_parts(content_parts)
            )
        else:
            plan = self._planning_call_with_retry(
                lambda: self._call_gemini_json(planning_prompt)
            )
        if plan is None or self._cancel_event.is_set():
            return None

        steps = plan.get("steps", [])
        goal_summary = plan.get("goal_summary", user_prompt)

        step_summaries = [
            {
                "effect_name": s.get("effect_name", ""),
                "operation": s.get("operation", "add"),
                "rationale": s.get("rationale", ""),
            }
            for s in steps
        ]

        self.ui_queue.put(
            {
                "type": "plan_ready",
                "goal_summary": goal_summary,
                "step_count": len(steps),
                "steps": step_summaries,
            }
        )

        return steps, goal_summary

    def _execute_summary_phase(
        self,
        user_prompt: str,
        goal_summary: str,
        applied_steps: list[tuple[str, str]],
    ) -> None:
        """Run the summary phase common to both modes."""
        self._conversation_history.append(f"User: {user_prompt}")
        self._conversation_history.append(f"Agent: {goal_summary}")

        if applied_steps:
            steps_str = "\n".join(
                f"- {name} ({label})" for name, label in applied_steps
            )
            summary_prompt = (
                f'The user asked: "{user_prompt}"\n'
                f"The editing goal was: {goal_summary}\n"
                f"The following edits were applied to the photo:\n{steps_str}\n\n"
                "Write a single short sentence (max 12 words) summarizing the edit. "
                "Be direct and specific. No filler words."
            )
            summary = self._call_gemini_text(summary_prompt) or (
                "Done. " + ", ".join(f"{n} ({l})" for n, l in applied_steps) + "."
            )
        else:
            summary = "Done. No edits were applied."

        self.ui_queue.put({"type": "done", "summary": summary})

    def _describe_stack(self, stack_snapshot: list[dict]) -> str:
        """Format the effects stack as a compact human-readable string."""
        if not stack_snapshot:
            return "Empty (no effects applied)"
        lines = []
        for i, item in enumerate(stack_snapshot):
            effect_class = item.get("effect", "Unknown")
            state = item.get("state", {})
            param_strs = []
            for k, v in state.items():
                param_strs.append(f"{k}={_fmt_stack_value(v)}")
            params_str = ", ".join(param_strs) if param_strs else "default"
            lines.append(f"  {i}: {effect_class}({params_str})")
        return "\n".join(lines)

    @staticmethod
    def _extract_text(response) -> str:
        """Extract only text parts from a response, ignoring thought_signature parts."""
        parts = response.candidates[0].content.parts
        return "".join(p.text for p in parts if hasattr(p, "text") and p.text)

    def _call_gemini_text(self, prompt: str) -> str | None:
        """Call Gemini with a text prompt expecting a plain text response."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return self._extract_text(response).strip()
        except Exception as exc:
            self._handle_api_error(exc, "Agent text error")
            return None

    def _call_gemini_json(self, prompt: str) -> dict | None:
        """Call Gemini with a text prompt expecting a JSON response."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            parsed = json.loads(self._extract_text(response))
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:
            self._handle_api_error(exc, "Agent planning error")
            return None

    def _call_gemini_json_with_parts(self, content_parts: list) -> dict | None:
        """Call Gemini with mixed text + image parts expecting a JSON response."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            parsed = json.loads(self._extract_text(response))
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:
            self._handle_api_error(exc, "Agent vision error")
            return None

    def _handle_api_error(self, exc: Exception, prefix: str) -> None:
        """Push an error message to the UI queue, listing available models on 404."""
        msg = str(exc)
        if "404" in msg or "not found" in msg.lower():
            try:
                available = [m.name for m in self.client.models.list()]
                models_str = "\n  ".join(available) if available else "(none returned)"
                msg = (
                    f"Model '{self.model}' not found (404).\n"
                    f"Available models:\n  {models_str}"
                )
            except Exception:
                msg = f"Model '{self.model}' not found (404). Could not list available models."
        self.ui_queue.put({"type": "error", "text": f"{prefix}: {msg}"})

    def _filter_effects_list(self, effects_list: str) -> str:
        """Remove excluded effects from the effects list string."""
        lines = [
            line
            for line in effects_list.split("\n")
            if not any(f"- {ex}" in line for ex in self._EXCLUDED_EFFECTS)
        ]
        return "\n".join(lines)

    def _filter_params_catalog(self, catalog: str) -> str:
        """Remove excluded effects' sections from the params catalog string."""
        lines = catalog.split("\n")
        result = []
        skip = False
        for line in lines:
            if line.startswith("Effect key"):
                m = re.search(r"\(([^)]+)\)", line)
                if m:
                    keys = {k.strip() for k in m.group(1).split(",")}
                    skip = bool(keys & self._EXCLUDED_EFFECTS)
                else:
                    skip = False
            if not skip:
                result.append(line)
        return "\n".join(result)

    def _describe_effect_params(self, effect_name: str) -> str:
        """Format the full parameter catalog for a single effect."""
        try:
            instance = self.effect_registry[effect_name]()
            params = instance.get_params()
        except Exception:
            return "No parameters available"
        lines = []
        for p in params:
            if p.get("type") == "button":
                continue
            ptype = p.get("type", "unknown")
            pmin, pmax, default = p.get("min"), p.get("max"), p.get("default")
            for attr in ("default", "min", "max"):
                val = p.get(attr)
                if hasattr(val, "tolist"):
                    p = {**p, attr: val.tolist()}
            pmin, pmax, default = p.get("min"), p.get("max"), p.get("default")
            parts = [f"  - {p['name']} ({ptype})"]
            if pmin is not None and pmax is not None:
                parts.append(f"range: {pmin} to {pmax}")
            if default is not None:
                parts.append(f"default: {default}")
            lines.append(", ".join(parts))
        return "\n".join(lines) if lines else "No parameters available"

    def _refine_params(
        self,
        step: dict,
        chosen_label: str,
        current_params: dict,
        thumbnail: torch.Tensor,
        stack_index: int | None,
        goal_summary: str,
    ) -> tuple[torch.Tensor, dict]:
        """Ask Gemini to fine-tune the just-applied effect if the preset isn't strong enough.

        Returns (updated_thumbnail, updated_params). If satisfied, both are unchanged.
        """
        effect_name = step.get("effect_name", "")
        if effect_name not in self.effect_registry or stack_index is None:
            return thumbnail, current_params

        current_params_str = "\n".join(
            f"  {k}: {_fmt_stack_value(v)}" for k, v in current_params.items()
        )
        effect_params_str = self._describe_effect_params(effect_name)

        refinement_prompt = _safe_format(
            self._refinement_prompt_template,
            goal_summary=goal_summary,
            effect_name=effect_name,
            chosen_label=chosen_label,
            current_params=current_params_str,
            effect_params=effect_params_str,
        )

        jpeg = tensor_to_vision_jpeg(thumbnail)
        content_parts: list[Any] = [
            refinement_prompt,
            types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
        ]

        result = self._call_gemini_json_with_parts(content_parts)
        if result is None or result.get("satisfied", True):
            return thumbnail, current_params

        adjusted = result.get("adjusted_params") or {}
        if not adjusted:
            return thumbnail, current_params

        try:
            eff = self._instantiate_effect(
                effect_name, adjusted, base_params=current_params
            )
            if eff is None:
                return thumbnail, current_params

            refined_params = self._extract_params(eff)
            reasoning = result.get("reasoning", "adjusted parameters")

            self.ui_queue.put(
                {
                    "type": "apply_effect",
                    "operation": "modify",
                    "effect_name": effect_name,
                    "params": refined_params,
                    "stack_index": stack_index,
                }
            )
            self.agent_queue.get()

            new_thumbnail = self._apply_to_thumbnail(eff, thumbnail)
            self.ui_queue.put(
                {
                    "type": "status",
                    "text": f"Refined {effect_name}: {reasoning}",
                }
            )
            return new_thumbnail, refined_params

        except Exception:
            return thumbnail, current_params

    def _augment_params_catalog(
        self, base_catalog: str, registry: dict | None = None
    ) -> str:
        """Append text-type parameters to the params catalog.

        GeminiActionMatcher excludes text params (search_query etc.) from the
        catalog it builds because they are not suitable for the slider preset
        workflow.  The agentic planner must see them so Gemini knows to include
        a search_query value when planning a SearchLUT step.
        """
        if registry is None:
            registry = self.effect_registry
        text_lines: list[str] = []
        seen_classes: set[str] = set()
        for key in sorted(registry.keys()):
            try:
                inst = registry[key]()
                cls_name = type(inst).__name__
                if cls_name in seen_classes:
                    continue
                seen_classes.add(cls_name)
                text_params = [p for p in inst.get_params() if p.get("type") == "text"]
                if not text_params:
                    continue
                text_lines.append(f"Effect key ({key}):")
                for p in text_params:
                    default = p.get("default", "")
                    text_lines.append(
                        f'  - {p["name"]} (text), default: "{default}"'
                        " -- set to a descriptive search phrase matching the desired look"
                    )
            except Exception:
                pass
        if not text_lines:
            return base_catalog
        return (
            base_catalog
            + "\n\nText search parameters (must be included in candidate_param_sets):\n"
            + "\n".join(text_lines)
        )

    def _load_prompt(self, filename: str) -> str:
        path = os.path.join(self._PROMPTS_DIR, filename)
        with open(path, "r") as f:
            return f.read()
