"""
Gemini-powered shader generation for text-to-shader effects.
"""

import json
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from google import genai
from src.utils.gemini import get_gemini_client


class GeminiShaderGenerator:
    """Generates GLSL shader code from text prompts using Gemini."""

    def __init__(self):
        self.client = get_gemini_client()
        self.model = "gemini-2.5-flash"  # Use Gemini Flash 2.5
        self._load_system_prompt()

    def _load_system_prompt(self) -> None:
        """Load the system prompt from the system_prompts directory."""
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "system_prompts",
            "t2s_shader_generation.txt",
        )
        with open(prompt_path, "r") as f:
            self.system_prompt_template = f.read()

    def _build_prompt(
        self, user_prompt: str, error_feedback: Optional[str] = None
    ) -> str:
        """
        Build the full prompt for Gemini shader generation.

        Args:
            user_prompt: The user's desired effect description
            error_feedback: Optional compilation error to fix

        Returns:
            Full prompt string for Gemini
        """
        base_prompt = self.system_prompt_template.format(user_prompt=user_prompt)

        if error_feedback:
            base_prompt += f"""

**COMPILATION ERROR FEEDBACK:**
The previous shader had this compilation error:
```
{error_feedback}
```

Please fix the shader code to resolve this error. Make sure to:
1. Check GLSL syntax carefully
2. Ensure all variables are declared
3. Verify function signatures match usage
4. Keep parameters alphabetically sorted
"""

        return base_prompt

    def generate_shader(
        self,
        prompt: str,
        max_retries: int = 3,
        error_feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate shader code from a text prompt.

        Args:
            prompt: User's desired effect description
            max_retries: Maximum retry attempts on failure
            error_feedback: Optional compilation error from previous attempt

        Returns:
            Dictionary with keys:
                - shader_code: GLSL function code (str)
                - parameters: List of parameter definitions (list of dicts)
                - description: Effect description (str)
                - success: Whether generation succeeded (bool)
                - error: Error message if failed (str, optional)

        Raises:
            Exception: If all retries are exhausted
        """
        full_prompt = self._build_prompt(prompt, error_feedback)

        for attempt in range(max_retries):
            try:
                print(
                    f"🤖 Generating shader with Gemini (attempt {attempt + 1}/{max_retries})..."
                )

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                )

                # Extract JSON from response
                response_text = response.text.strip()

                # Remove markdown code fences if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]

                response_text = response_text.strip()

                # Parse JSON
                shader_dict = json.loads(response_text)

                # Validate required fields
                required_fields = ["shader_code", "parameters", "description"]
                for field in required_fields:
                    if field not in shader_dict:
                        raise ValueError(f"Missing required field: {field}")

                # Verify parameters are alphabetically sorted
                param_names = [p["name"] for p in shader_dict["parameters"]]
                if param_names != sorted(param_names):
                    print(
                        f"⚠️  Warning: Parameters not alphabetically sorted. Auto-fixing..."
                    )
                    shader_dict["parameters"].sort(key=lambda p: p["name"])
                    # Note: This doesn't fix the function signature - validator will catch it

                shader_dict["success"] = True
                print(f"✅ Shader generated successfully!")
                return shader_dict

            except json.JSONDecodeError as e:
                error_msg = f"JSON parsing error: {e}"
                print(f"❌ {error_msg}")
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "error": error_msg,
                        "raw_response": response_text
                        if "response_text" in locals()
                        else None,
                    }

            except Exception as e:
                error_msg = f"Generation error: {e}"
                print(f"❌ {error_msg}")

                # Check if it's a rate limit error
                error_str = str(e).lower()
                is_rate_limit = any(
                    keyword in error_str
                    for keyword in [
                        "rate",
                        "quota",
                        "limit",
                        "429",
                        "resource_exhausted",
                        "resourceexhausted",
                    ]
                )

                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"⚠️  Rate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                if attempt == max_retries - 1:
                    return {"success": False, "error": error_msg}

        return {
            "success": False,
            "error": "Max retries exhausted",
        }

    def get_passthrough_shader(self) -> Dict[str, Any]:
        """
        Get a simple passthrough shader (identity function) as fallback.

        Returns:
            Shader dictionary for passthrough effect
        """
        return {
            "shader_code": """vec4 apply_t2s(vec4 color) {
    // Passthrough - generation failed
    return color;
}""",
            "parameters": [],
            "description": "Passthrough (generation failed)",
            "success": True,
        }
