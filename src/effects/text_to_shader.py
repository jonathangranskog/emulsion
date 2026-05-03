"""
Text-to-Shader (T2S) effect that generates GLSL shaders from text prompts using Gemini.
"""

from typing import List, Dict, Any, Optional
import torch
from src.effects.base import ImageEffect
from src.generation.gemini_shader import GeminiShaderGenerator
from src.generation.shader_validator import ShaderValidator


class TextToShaderEffect(ImageEffect):
    """
    Dynamically generates shader code from text prompts using Gemini.

    Users can type "T2S: <description>" to create custom effects without coding.
    """

    def __init__(self, prompt: str = "", cache_manager=None):
        super().__init__()
        self.prompt = prompt
        self.cache_manager = cache_manager  # Will be set by cache system

        # Generation state
        self.generation_status = "pending"  # pending, generating, success, error
        self.generation_attempted = False

        # Generated shader data
        self.shader_code: Optional[str] = None
        self.shader_parameters: List[Dict[str, Any]] = []
        self.shader_description: str = ""
        self.error_message: str = ""

        # Dynamic parameter values (set from generated parameters)
        self._dynamic_param_values: Dict[str, Any] = {}

        # Generator and validator
        self._generator = None
        self._validator = None

    @property
    def generator(self) -> GeminiShaderGenerator:
        """Lazy initialization of generator."""
        if self._generator is None:
            self._generator = GeminiShaderGenerator()
        return self._generator

    @property
    def validator(self) -> ShaderValidator:
        """Lazy initialization of validator."""
        if self._validator is None:
            self._validator = ShaderValidator()
        return self._validator

    def _generate_shader(self) -> bool:
        """
        Generate shader from prompt using Gemini.

        Returns:
            True if generation succeeded, False otherwise
        """
        if not self.prompt or self.prompt.strip() == "":
            self.generation_status = "error"
            self.error_message = "Empty prompt"
            return False

        self.generation_status = "generating"
        print(f"🎨 Generating shader for prompt: '{self.prompt}'")

        # Check cache first
        if self.cache_manager:
            cached_result = self.cache_manager.get_cached_shader(self.prompt)
            if cached_result:
                # RE-VALIDATE cached shaders to catch old broken ones
                print(f"🔍 Re-validating cached shader...")
                is_valid, validation_error = self.validator.validate_shader(
                    cached_result
                )

                if is_valid:
                    print(f"✅ Cached shader passed validation")
                    self._apply_shader_result(cached_result)
                    self.generation_status = "success"
                    return True
                else:
                    print(f"❌ Cached shader failed validation: {validation_error}")
                    print(f"   Regenerating from scratch...")
                    # Fall through to generate new shader

        # Generate with Gemini
        max_attempts = 3
        error_feedback = None

        for attempt in range(max_attempts):
            print(f"📝 Generation attempt {attempt + 1}/{max_attempts}")

            result = self.generator.generate_shader(
                self.prompt, max_retries=2, error_feedback=error_feedback
            )

            if not result.get("success", False):
                self.error_message = result.get("error", "Unknown error")
                print(f"❌ Generation failed: {self.error_message}")

                if attempt == max_attempts - 1:
                    # Final attempt failed - use passthrough
                    print(
                        "⚠️  All generation attempts failed. Using passthrough shader."
                    )
                    result = self.generator.get_passthrough_shader()
                    self._apply_shader_result(result)
                    self.generation_status = "error"
                    return False

                continue

            # Validate the generated shader
            is_valid, validation_error = self.validator.validate_shader(result)

            if not is_valid:
                print(f"⚠️  Validation failed: {validation_error}")

                # Try to auto-fix parameter ordering
                fixed, fixed_result, fix_msg = self.validator.auto_fix_parameter_order(
                    result
                )

                if fixed:
                    print(f"🔧 Auto-fixed: {fix_msg}")
                    result = fixed_result
                    is_valid = True
                else:
                    # Send validation error back to Gemini for fixing
                    print(f"🔄 Sending error back to Gemini for fixing...")

                    # Make error message more specific for Gemini
                    if "texture" in validation_error.lower():
                        error_feedback = (
                            f"{validation_error}\n\n"
                            "CRITICAL: You MUST use 'main_texture' (uniform sampler2D) "
                            "for texture sampling, NOT 'texture0', 'texture1', or any other name. "
                            "Example: texture(main_texture, TexCoord + offset).rgb"
                        )
                    elif (
                        "coord" in validation_error.lower()
                        or "uv" in validation_error.lower()
                    ):
                        error_feedback = (
                            f"{validation_error}\n\n"
                            "CRITICAL: Use 'TexCoord' (vec2) for texture coordinates, "
                            "NOT 'uv', 'texCoord', or 'coord'."
                        )
                    elif "textureSize" in validation_error.lower():
                        error_feedback = (
                            f"{validation_error}\n\n"
                            "CRITICAL: textureSize() is NOT available. "
                            "Use fixed pixel offsets like vec2(0.001, 0.0) or vec2(0.002, 0.002)."
                        )
                    else:
                        error_feedback = validation_error

                    continue

            # Success!
            print(f"✅ Shader validated successfully!")
            self._apply_shader_result(result)
            self.generation_status = "success"

            # Cache the successful result
            if self.cache_manager:
                self.cache_manager.cache_shader(self.prompt, result)

            return True

        # Should not reach here, but just in case
        self.generation_status = "error"
        return False

    def _apply_shader_result(self, result: Dict[str, Any]):
        """Apply a generated/cached shader result to this effect."""
        self.shader_code = result["shader_code"]
        self.shader_parameters = result["parameters"]
        self.shader_description = result.get("description", "")

        # Initialize dynamic parameter values with defaults
        for param in self.shader_parameters:
            param_name = param["name"]
            param_default = param.get("default")

            # Convert list defaults to torch tensors for vec types
            if isinstance(param_default, list):
                param_default = torch.tensor(param_default, dtype=torch.float32)

            self._dynamic_param_values[param_name] = param_default
            # Also set as attribute for easy access
            setattr(self, param_name, param_default)

    def poll_generation_status(self) -> Dict[str, Any] | None:
        if self.generation_status == "success":
            return {"status": "success"}
        if self.generation_status == "error":
            return {"status": "error", "message": self.error_message}
        if self.generation_status == "generating":
            return {"status": "generating"}
        return None

    def get_params(self) -> List[Dict[str, Any]]:
        """
        Return parameter definitions.

        For T2S effects, parameters are dynamically generated based on the shader.
        """
        params = []

        # Always show the prompt (read-only text field)
        params.append(
            {
                "name": "prompt",
                "label": "Prompt",
                "type": "text",
                "default": self.prompt,
                "readonly": True,  # User shouldn't edit this after creation
            }
        )

        # Add generation status indicator
        status_label = {
            "pending": "⏳ Pending",
            "generating": "🔄 Generating...",
            "success": "✅ Success",
            "error": "❌ Error",
        }.get(self.generation_status, "Unknown")

        params.append(
            {
                "name": "generation_status",
                "label": "Status",
                "type": "text",
                "default": status_label,
                "readonly": True,
            }
        )

        # If generation succeeded, show the generated parameters
        if self.generation_status == "success" and self.shader_parameters:
            for param_def in self.shader_parameters:
                params.append(param_def)

        # If generation failed, show error message
        if self.generation_status == "error":
            params.append(
                {
                    "name": "error_message",
                    "label": "Error",
                    "type": "text",
                    "default": self.error_message,
                    "readonly": True,
                }
            )

        return params

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        CPU processing (not used for T2S - GPU only).

        Returns input unchanged since T2S effects run entirely on GPU.
        """
        return x

    def get_shader_info(self) -> tuple[list[str], Dict[str, Any]]:
        """
        Get shader code and uniforms for GPU processing.

        This triggers generation on first call if not already generated.

        IMPORTANT: This method must NEVER raise exceptions, as that would crash
        the shader compilation pipeline. Always return a safe passthrough shader
        if anything goes wrong.
        """
        try:
            # Generate shader if not done yet
            if not self.generation_attempted:
                self.generation_attempted = True
                self._generate_shader()

            # If generation failed or not yet complete, return passthrough
            if self.generation_status != "success" or not self.shader_code:
                passthrough = """
vec4 apply_t2s(vec4 color) {
    return color;
}
"""
                return [passthrough], {}

            # Build uniforms from current parameter values
            uniforms = {}
            for param in self.shader_parameters:
                param_name = param["name"]
                uniform_name = f"u_t2s_{param_name}"

                # Get current value (might have been updated by UI)
                value = getattr(
                    self, param_name, self._dynamic_param_values.get(param_name)
                )

                # Ensure vec types are tensors
                if param["type"] in ["vec3", "vec4"] and not isinstance(
                    value, torch.Tensor
                ):
                    value = torch.tensor(value, dtype=torch.float32)

                uniforms[uniform_name] = value

            return [self.shader_code], uniforms

        except Exception as e:
            # CRITICAL: Never crash the shader pipeline
            # Log the error and return a safe passthrough
            print(f"❌ CRITICAL ERROR in TextToShaderEffect.get_shader_info(): {e}")
            print(f"   Returning passthrough shader to prevent crash")

            # Mark as error for UI feedback
            self.generation_status = "error"
            self.error_message = f"Runtime error: {e}"

            # Return safe passthrough
            passthrough = """
vec4 apply_t2s(vec4 color) {
    return color;
}
"""
            return [passthrough], {}

    def get_glsl_prefix(self) -> str:
        """Return the prefix for the shader function (used in compilation)."""
        return "t2s"

    def serialize_to_cache(self) -> Dict[str, Any]:
        """
        Serialize effect state for caching/undo.

        Returns:
            Dictionary with all state needed to restore the effect
        """
        state = {
            "prompt": self.prompt,
            "generation_status": self.generation_status,
            "generation_attempted": self.generation_attempted,
            "shader_code": self.shader_code,
            "shader_parameters": self.shader_parameters,
            "shader_description": self.shader_description,
            "error_message": self.error_message,
            "toggled": self.toggled,
        }

        # Include current parameter values
        param_values = {}
        for param in self.shader_parameters:
            param_name = param["name"]
            value = getattr(self, param_name, None)
            # Convert tensors to lists for JSON serialization
            if isinstance(value, torch.Tensor):
                value = value.tolist()
            param_values[param_name] = value

        state["parameter_values"] = param_values

        return state

    @classmethod
    def deserialize_from_cache(cls, state: Dict[str, Any]) -> "TextToShaderEffect":
        """
        Restore effect from cached state.

        Args:
            state: Dictionary from serialize_to_cache()

        Returns:
            Restored TextToShaderEffect instance
        """
        # Import cache manager here to avoid circular imports
        from src.generation.shader_cache import ShaderCacheManager

        cache_manager = ShaderCacheManager()
        effect = cls(prompt=state["prompt"], cache_manager=cache_manager)

        effect.generation_status = state.get("generation_status", "pending")
        effect.generation_attempted = state.get("generation_attempted", True)
        effect.shader_code = state.get("shader_code")
        effect.shader_parameters = state.get("shader_parameters", [])
        effect.shader_description = state.get("shader_description", "")
        effect.error_message = state.get("error_message", "")
        effect.toggled = state.get("toggled", True)

        # Restore parameter values
        param_values = state.get("parameter_values", {})
        for param_name, value in param_values.items():
            # Convert lists back to tensors for vec types
            if isinstance(value, list):
                value = torch.tensor(value, dtype=torch.float32)
            effect._dynamic_param_values[param_name] = value
            setattr(effect, param_name, value)

        return effect
