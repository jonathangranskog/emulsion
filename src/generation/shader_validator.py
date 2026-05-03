"""
Validation and compilation testing for generated shaders.
"""

import re
from typing import Dict, List, Any, Tuple, Optional
import OpenGL.GL as gl
from src.interface.shader import compile_shader_program, PASSTHROUGH_VERTEX_SHADER
from src.interface.uniform import infer_glsl_type


class ShaderValidator:
    """Validates and test-compiles generated shader code."""

    def __init__(self):
        pass

    def extract_function_parameters(self, shader_code: str) -> List[str]:
        """
        Extract parameter names from the shader function signature.

        Args:
            shader_code: Full shader code containing the function

        Returns:
            List of parameter names in order (excluding the first 'color' param)

        Raises:
            ValueError: If function signature cannot be parsed
        """
        # Match: vec4 apply_t2s(vec4 color, type1 param1, type2 param2, ...)
        pattern = r"vec4\s+apply_t2s\s*\(\s*vec4\s+color\s*(?:,\s*\w+\s+(\w+)\s*)*\)"

        # Find the full function signature
        match = re.search(
            r"vec4\s+apply_t2s\s*\([^)]+\)",
            shader_code,
            re.MULTILINE | re.DOTALL,
        )

        if not match:
            raise ValueError("Could not find 'vec4 apply_t2s(...)' function signature")

        signature = match.group(0)

        # Extract all parameters after 'vec4 color'
        # Match patterns like: float brightness, vec3 tint_color, etc.
        param_pattern = r",\s*(?:float|int|vec[234])\s+(\w+)"
        params = re.findall(param_pattern, signature)

        return params

    def verify_alphabetical_order(
        self, shader_code: str, parameter_defs: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify that parameters are alphabetically sorted in both the function
        signature and the parameter definitions list.

        Args:
            shader_code: The generated shader code
            parameter_defs: List of parameter definition dicts

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Extract parameters from function signature
            signature_params = self.extract_function_parameters(shader_code)

            # Extract parameters from definitions
            definition_params = [p["name"] for p in parameter_defs]

            # Check if signature params are sorted
            signature_sorted = signature_params == sorted(signature_params)

            # Check if definition params are sorted
            definitions_sorted = definition_params == sorted(definition_params)

            # Check if they match each other
            params_match = signature_params == definition_params

            if not signature_sorted:
                return (
                    False,
                    f"Function signature parameters are not alphabetically sorted. "
                    f"Found: {signature_params}, Expected: {sorted(signature_params)}",
                )

            if not definitions_sorted:
                return (
                    False,
                    f"Parameter definitions are not alphabetically sorted. "
                    f"Found: {definition_params}, Expected: {sorted(definition_params)}",
                )

            if not params_match:
                return (
                    False,
                    f"Function signature parameters don't match definitions. "
                    f"Signature: {signature_params}, Definitions: {definition_params}",
                )

            return True, None

        except ValueError as e:
            return False, str(e)

    def test_compile_shader(
        self, shader_code: str, uniforms: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Test-compile the shader with dummy uniforms to catch syntax errors.

        Args:
            shader_code: The GLSL shader code
            uniforms: Dictionary of uniform name -> value

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Build uniform declarations
            uniform_declarations = []
            for uniform_name, uniform_value in uniforms.items():
                glsl_type = infer_glsl_type(uniform_value)
                uniform_declarations.append(f"uniform {glsl_type} {uniform_name};")

            # Build argument list (sorted alphabetically, matching shader_processor)
            arg_list = ", ".join(sorted(uniforms.keys())) if uniforms else ""
            function_call = f"apply_t2s(color{', ' + arg_list if arg_list else ''})"

            # Build full fragment shader (matching shader_processor format)
            fragment_source = f"""
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D main_texture;

{chr(10).join(uniform_declarations)}

{shader_code}

void main() {{
    vec4 color = texture(main_texture, TexCoord);
    color = {function_call};
    FragColor = color;
}}
"""

            # Attempt compilation
            program = compile_shader_program(PASSTHROUGH_VERTEX_SHADER, fragment_source)

            # Clean up the test program
            gl.glDeleteProgram(program)

            return True, None

        except RuntimeError as e:
            error_msg = str(e)
            return False, error_msg

        except Exception as e:
            return False, f"Unexpected error during compilation: {e}"

    def check_forbidden_patterns(self, shader_code: str) -> Tuple[bool, Optional[str]]:
        """
        Check for common errors and forbidden patterns in shader code.

        Args:
            shader_code: The GLSL shader code

        Returns:
            Tuple of (is_valid, error_message)
        """
        import re

        # Check for incorrect texture names (common error)
        forbidden_textures = [
            r"\btexture0\b",
            r"\btexture1\b",
            r"\binputTexture\b",
            r"\buTexture\b",
            r"\bsamplerTex\b",
            r"\btex\b(?!\w)",  # 'tex' but not 'texture'
        ]

        for pattern in forbidden_textures:
            if re.search(pattern, shader_code):
                match = re.search(pattern, shader_code).group(0)
                return (
                    False,
                    f"Incorrect texture name '{match}' found. Use 'main_texture' instead.",
                )

        # Check for textureSize() which is not available
        if re.search(r"\btextureSize\s*\(", shader_code):
            return (
                False,
                "textureSize() is not available. Use fixed pixel offsets instead (e.g., vec2(0.001, 0.0)).",
            )

        # Check for undefined variable names that look like texture coordinates
        wrong_coord_names = [r"\buv\b", r"\btexCoord\b", r"\bcoord\b"]
        for pattern in wrong_coord_names:
            if re.search(pattern, shader_code):
                match = re.search(pattern, shader_code).group(0)
                return (
                    False,
                    f"Incorrect coordinate variable '{match}'. Use 'TexCoord' instead.",
                )

        return True, None

    def validate_shader(
        self, shader_dict: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Perform full validation on a generated shader dictionary.

        Args:
            shader_dict: Generated shader dictionary with keys:
                - shader_code: GLSL function code
                - parameters: List of parameter definitions

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        if "shader_code" not in shader_dict:
            return False, "Missing 'shader_code' field"

        if "parameters" not in shader_dict:
            return False, "Missing 'parameters' field"

        shader_code = shader_dict["shader_code"]
        parameters = shader_dict["parameters"]

        # Check for forbidden patterns FIRST (before compilation)
        is_valid_pattern, pattern_error = self.check_forbidden_patterns(shader_code)
        if not is_valid_pattern:
            return False, pattern_error

        # Validate alphabetical ordering
        is_sorted, sort_error = self.verify_alphabetical_order(shader_code, parameters)
        if not is_sorted:
            return False, sort_error

        # Build dummy uniforms for test compilation
        uniforms = {}
        for param in parameters:
            param_name = param["name"]
            param_type = param["type"]
            param_default = param.get("default", 0.0)

            # Create uniform name (add u_ prefix to match convention)
            uniform_name = f"u_t2s_{param_name}"
            uniforms[uniform_name] = param_default

        # Test compile the shader
        compiled, compile_error = self.test_compile_shader(shader_code, uniforms)
        if not compiled:
            return False, f"Shader compilation failed: {compile_error}"

        return True, None

    def auto_fix_parameter_order(
        self, shader_dict: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Attempt to automatically fix parameter ordering issues.

        This sorts the parameter definitions, but cannot fix the function signature
        (which requires regeneration by Gemini).

        Args:
            shader_dict: Generated shader dictionary

        Returns:
            Tuple of (fixed_successfully, updated_dict, message)
        """
        try:
            # Sort parameter definitions alphabetically
            shader_dict["parameters"].sort(key=lambda p: p["name"])

            # Try to extract and validate function signature parameters
            signature_params = self.extract_function_parameters(
                shader_dict["shader_code"]
            )
            definition_params = [p["name"] for p in shader_dict["parameters"]]

            # If signature is still not sorted, we can't auto-fix (needs regeneration)
            if signature_params != sorted(signature_params):
                return (
                    False,
                    shader_dict,
                    "Cannot auto-fix: function signature needs regeneration",
                )

            # If parameters don't match, we can't auto-fix
            if signature_params != definition_params:
                return (
                    False,
                    shader_dict,
                    f"Parameter mismatch: signature has {signature_params}, "
                    f"definitions have {definition_params}",
                )

            return True, shader_dict, "Parameter definitions sorted successfully"

        except Exception as e:
            return False, shader_dict, f"Auto-fix failed: {e}"
