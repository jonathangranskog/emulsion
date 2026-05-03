import OpenGL.GL as gl
import torch
import numpy as np

from src.core.image import ImageData
from src.pipeline.stack import EffectStack
from src.pipeline.state_cache import EffectStateCache
from src.effects.base import ImageEffect
from src.effects.crop import Crop
from src.interface.texture import TextureManager
from src.interface.uniform import infer_glsl_type, upload_uniform
from src.interface.shader import compile_shader_program, get_passthrough_shader_program
from src.interface.geometry import get_full_screen_quad_vao


# Simple vertex shader for fullscreen quad (no MVP transforms needed)
FULLSCREEN_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
""".strip()

# The shader that is used to render the image to screen from FBO
FINAL_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D main_texture;

void main() {
    FragColor = texture(main_texture, vec2(TexCoord.x, 1.0 - TexCoord.y));
}
""".strip()


class PingPongFBO:
    """Manages two framebuffers for ping-pong rendering between effects"""

    def __init__(self, width: int, height: int, name: str = None):
        self.width = width
        self.height = height

        # Create unique names based on dimensions if not provided
        if name is None:
            name = f"ping_pong_{width}x{height}"

        self.fbo_a = f"{name}_fbo_a"
        self.fbo_b = f"{name}_fbo_b"
        self.texture_a = f"{name}_texture_a"
        self.texture_b = f"{name}_texture_b"

        TextureManager.register_fbo(self.fbo_a, self.texture_a, width, height)
        TextureManager.register_fbo(self.fbo_b, self.texture_b, width, height)
        self.current_read = self.fbo_a
        self.current_write = self.fbo_b

    def swap(self):
        """Swap read/write buffers"""
        self.current_read, self.current_write = self.current_write, self.current_read

    def get_read_texture_id(self) -> int:
        """Get the texture ID of the current read buffer"""
        texture_name = TextureManager.get_fbo_texture_name(self.current_read)
        return TextureManager.get_texture_id(texture_name)

    def get_read_texture_name(self) -> str:
        """Get the texture name of the current read buffer"""
        return TextureManager.get_fbo_texture_name(self.current_read)

    def get_write_fbo_id(self) -> int:
        """Get the FBO ID of the current write buffer"""
        return TextureManager.get_fbo_id(self.current_write)

    def cleanup(self):
        """Delete OpenGL resources"""
        TextureManager.delete_fbo(self.fbo_a)
        TextureManager.delete_fbo(self.fbo_b)


class ShaderEffectsProcessor:
    def __init__(self, source: ImageData, stack: EffectStack):
        self.source = source
        self.stack = stack
        self.effect_programs: list[list[int]] = []
        self.fbo_pool = {}  # Will be populated in compile_effect_shaders()
        self.texture_name_uniform_map: list[dict[str, str]] = []
        self.create_fbo_pool()
        self.compile_effect_shaders()
        self.upload_effect_textures()
        self.passthrough_program = get_passthrough_shader_program()
        self.quad_vao = get_full_screen_quad_vao()
        self.source_texture_id = TextureManager.get_texture_id("main_texture")

        # Bypass state: dedicated FBO for crop-only output (separate from main pool)
        self._bypass_fbo: PingPongFBO | None = None
        self._bypass_width: int = 0
        self._bypass_height: int = 0
        self._needs_full_reprocess: bool = False

    def reconstruction_required(self) -> bool:
        return self.stack.reconstruction_required()

    def values_changed(self) -> bool:
        return self.stack.values_changed()

    def on_frame(self, texture):
        # After unbypass, trigger full reprocess if params changed during bypass
        if self._needs_full_reprocess and not self.stack.effects_bypassed:
            self._needs_full_reprocess = False
            self.stack.mark_values_changed()

        # When bypassed, handle structural changes but skip effect processing
        if self.stack.effects_bypassed:
            if self.reconstruction_required() or self.values_changed():
                if self.reconstruction_required():
                    original_viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
                    self.create_fbo_pool()
                    EffectStateCache.set_current_source(self.source.tensor)
                    self.compile_effect_shaders()
                    self.upload_effect_textures()
                    self.stack.clear_reconstruction_required()
                    gl.glViewport(
                        original_viewport[0],
                        original_viewport[1],
                        original_viewport[2],
                        original_viewport[3],
                    )
                # Re-prepare bypass output (e.g. crop params changed)
                self._render_bypass_output()
                self.stack.clear_values_changed()
                self._needs_full_reprocess = True
            return

        # Only process if something changed
        if not (self.reconstruction_required() or self.values_changed()):
            return

        # Save current viewport to restore later
        original_viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)

        # Recompile shaders if stack changed
        if self.reconstruction_required():
            self.create_fbo_pool()
            EffectStateCache.set_current_source(self.source.tensor)
            self.compile_effect_shaders()
            self.upload_effect_textures()
            self.stack.clear_reconstruction_required()

        # Track current dimensions and input texture through the pipeline
        resolution = TextureManager.get_texture_resolution("main_texture")
        current_width, current_height = resolution[1], resolution[0]
        current_input_texture = self.source_texture_id

        # Initialize state cache for this frame
        EffectStateCache.begin_update(self.stack.effects)
        EffectStateCache.set_source_shape((3, current_height, current_width))

        # Start by copying source to the base FBO
        current_fbo = self.fbo_pool[(current_width, current_height)]
        self._copy_texture_to_fbo(
            current_input_texture,
            current_fbo.get_write_fbo_id(),
            current_width,
            current_height,
        )
        current_fbo.swap()
        current_input_texture = current_fbo.get_read_texture_id()
        current_input_texture_name = current_fbo.get_read_texture_name()

        # Apply each effect
        for i, effect in enumerate(self.stack.effects):
            if not effect.toggled:
                continue

            # Cache intermediate state for effects that need it (via FBO read)
            if effect.requires_intermediate_state:
                input_tensor = self._read_texture_to_tensor(
                    current_input_texture, current_width, current_height
                )
                EffectStateCache.set_intermediate_state(effect, input_tensor)

            # Calculate output dimensions for this effect
            delta_w, delta_h = self.effect_dimension_deltas[i]
            output_width = current_width + delta_w
            output_height = current_height + delta_h

            # Get the FBO for this output size
            output_fbo = self.fbo_pool[(output_width, output_height)]

            # First, render the entire FBO at full size with padding color
            gl.glViewport(0, 0, output_fbo.width, output_fbo.height)

            # Track effect's initial input for multi-pass effects
            effect_initial_input_texture = current_input_texture
            effect_initial_input_texture_name = current_input_texture_name

            # Special handling for effects with custom rendering
            programs = self.effect_programs[i]
            for program in programs:
                # Bind output FBO
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, output_fbo.get_write_fbo_id())

                # If effect has no shader code, use custom rendering
                if program is None:
                    effect.custom_render(
                        current_input_texture, current_width, current_height
                    )
                else:
                    # Otherwise render the shader program
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                    # Use effect's shader program
                    gl.glUseProgram(program)

                    # Bind input texture from the current read buffer to the effect's shader program
                    texture_unit = TextureManager.get_texture_unit(
                        current_input_texture_name
                    )
                    gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, current_input_texture)
                    gl.glUniform1i(
                        gl.glGetUniformLocation(program, "main_texture"), texture_unit
                    )

                    # Bind original texture if effect requires it (for multi-pass effects)
                    if effect.requires_original_texture():
                        original_texture_unit = TextureManager.get_texture_unit(
                            effect_initial_input_texture_name
                        )
                        gl.glActiveTexture(gl.GL_TEXTURE0 + original_texture_unit)
                        gl.glBindTexture(gl.GL_TEXTURE_2D, effect_initial_input_texture)
                        gl.glUniform1i(
                            gl.glGetUniformLocation(program, "original_texture"),
                            original_texture_unit,
                        )

                    # Upload effect uniforms
                    self._upload_effect_uniforms(program, effect, i)

                # Render fullscreen quad
                self._render_fullscreen_quad()

                # Swap buffers and update current state
                output_fbo.swap()
                current_input_texture = output_fbo.get_read_texture_id()
                current_input_texture_name = output_fbo.get_read_texture_name()
                current_width, current_height = output_width, output_height

        # Store final dimensions and FBO for output methods accounting for toggled effects
        self.final_width = current_width
        self.final_height = current_height
        self.final_fbo = self.fbo_pool[(current_width, current_height)]

        # Cache final output for preview system
        final_tensor = self._read_texture_to_tensor(
            current_input_texture, current_width, current_height
        )
        EffectStateCache.set_final_output(final_tensor)

        # Mark that values changed so viewer updates if needed
        self.stack.clear_values_changed()

        # Unbind FBO and reset state
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glUseProgram(0)

        # CRITICAL: Restore original viewport
        # Without this, the viewer renders with the huge FBO viewport (5584x8368)
        # instead of the window viewport (1280x720), causing massive scaling issues
        gl.glViewport(
            original_viewport[0],
            original_viewport[1],
            original_viewport[2],
            original_viewport[3],
        )

    def get_output_texture_id(self) -> int:
        """Get the final output texture ID for rendering to screen"""
        if self.stack.effects_bypassed:
            if self._bypass_fbo is not None:
                return self._bypass_fbo.get_read_texture_id()
            return self.source_texture_id
        return self.final_fbo.get_read_texture_id()

    def get_output_dimensions(self) -> tuple[int, int]:
        """Get the output dimensions (width, height) directly"""
        if self.stack.effects_bypassed:
            return (self._bypass_width, self._bypass_height)
        return (self.final_width, self.final_height)  # (W, H)

    def _read_texture_to_tensor(
        self, texture_id: int, width: int, height: int
    ) -> torch.Tensor:
        """Read a texture to a PyTorch tensor.

        Args:
            texture_id: OpenGL texture ID to read from
            width: Texture width
            height: Texture height

        Returns:
            Tensor with shape (C, H, W)
        """
        temp_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, temp_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_READ_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            texture_id,
            0,
        )

        pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_FLOAT)

        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
        gl.glDeleteFramebuffers(1, [temp_fbo])

        pixels_array = np.frombuffer(pixels, dtype=np.float32).reshape(height, width, 3)
        return torch.from_numpy(pixels_array).permute(2, 0, 1)

    def read_output_as_tensor(self) -> torch.Tensor:
        """Read the FBO output as a PyTorch tensor for saving

        Returns:
            Tensor with shape (C, H, W) in range [0, 1]
        """
        return self._read_texture_to_tensor(
            self.final_fbo.get_read_texture_id(),
            self.final_width,
            self.final_height,
        )

    def fragment_shader_string(self) -> str:
        """Return a simple passthrough shader for compatibility with viewer initialization"""
        return FINAL_FRAGMENT_SHADER

    def create_fbo_pool(self):
        """Create the FBO pool for the shader processor"""
        # Clean up old FBO pool
        for fbo in self.fbo_pool.values():
            fbo.cleanup()
        self.fbo_pool = {}

        # Get normalized dimension deltas per effect (independent of toggle state)
        normalized_deltas = [
            effect.get_effect_dimension_deltas() for effect in self.stack.effects
        ]

        # Pre-allocate FBOs for all possible sizes (assuming all effects enabled)
        resolution = TextureManager.get_texture_resolution("main_texture")
        current_width, current_height = resolution[1], resolution[0]

        all_sizes = {(current_width, current_height)}  # Start with original size

        # Calculate cumulative sizes through the effect chain
        # Convert normalized deltas to pixel deltas based on current dimensions
        self.effect_dimension_deltas = []
        for norm_delta_w, norm_delta_h in normalized_deltas:
            # Compute pixel deltas from normalized values using max dimension
            max_dim = max(current_width, current_height)
            delta_w_pixels = int(np.ceil(norm_delta_w * max_dim))
            delta_h_pixels = int(np.ceil(norm_delta_h * max_dim))

            self.effect_dimension_deltas.append((delta_w_pixels, delta_h_pixels))

            current_width += delta_w_pixels
            current_height += delta_h_pixels
            all_sizes.add((current_width, current_height))

        # Create PingPongFBOs for all unique sizes with unique names
        for width, height in all_sizes:
            name = f"fbo_{width}x{height}"
            self.fbo_pool[(width, height)] = PingPongFBO(width, height, name)

        self.final_width = current_width
        self.final_height = current_height
        self.final_fbo = self.fbo_pool[(current_width, current_height)]

    def compile_effect_shaders(self):
        """Compile individual shader programs for each effect"""
        # Clean up old programs
        for programs in self.effect_programs:
            for program in programs:
                if program:
                    gl.glDeleteProgram(program)
        self.effect_programs: list[list[int]] = []

        # Compile shaders for each effect
        for effect in self.stack.effects:
            glsl_snippets, uniforms = effect.get_shader_info()

            # Skip shader compilation for effects without shader code (e.g., Padding)
            this_effect_programs: list[int] = []
            for glsl_code in glsl_snippets:
                if not glsl_code or glsl_code.strip() == "":
                    this_effect_programs.append(None)
                else:
                    # Build fragment shader with uniforms
                    uniform_declarations = []
                    for uniform_name, uniform_value in uniforms.items():
                        glsl_type = infer_glsl_type(uniform_value)
                        uniform_declarations.append(
                            f"uniform {glsl_type} {uniform_name};"
                        )

                    # Build argument list for the effect function
                    arg_list = ", ".join(sorted(uniforms.keys())) if uniforms else ""
                    function_call = f"apply_{effect.get_glsl_prefix()}(color{', ' + arg_list if arg_list else ''})"

                    fragment_source = f"""#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D main_texture;
{"uniform sampler2D original_texture;" if effect.requires_original_texture() else ""}

{chr(10).join(uniform_declarations)}

{glsl_code}

void main() {{
    // Don't flip Y here - the viewer will handle coordinate flipping
    vec4 color = texture(main_texture, TexCoord);
    color = {function_call};
    FragColor = color;
}}
"""

                    program = compile_shader_program(
                        FULLSCREEN_VERTEX_SHADER, fragment_source
                    )
                    this_effect_programs.append(program)
            self.effect_programs.append(this_effect_programs)

    def upload_effect_textures(self):
        """Upload texture uniforms (like LUTs) to GPU"""
        # Delete all previous textures used by effects
        for effect_textures in self.texture_name_uniform_map:
            for texture_name in effect_textures.values():
                TextureManager.delete_texture(texture_name)
        self.texture_name_uniform_map = []
        used_texture_names = set[str]()

        # Loop through all effects and their uniforms to register textures
        for i, effect in enumerate(self.stack.effects):
            _, uniforms = effect.get_shader_info()
            effect_textures = {}

            for uniform_name, uniform_value in uniforms.items():
                glsl_type = infer_glsl_type(uniform_value)
                if glsl_type in ["sampler2D", "sampler3D"]:
                    assert isinstance(uniform_value, torch.Tensor), (
                        f"Uniform {uniform_name} is not a tensor"
                    )

                    # Rename to avoid conflicts with other effects
                    renamed_name = f"effect_{i}_{uniform_name}"
                    effect_textures[uniform_name] = renamed_name
                    assert renamed_name not in used_texture_names, (
                        f"Texture name {renamed_name} already used by another effect"
                    )
                    used_texture_names.add(renamed_name)

                    # Save the texture
                    if glsl_type == "sampler2D":
                        assert uniform_value.ndim == 3, (
                            f"Uniform {uniform_name} is not a 3D tensor"
                        )
                        TextureManager.register_texture2d(uniform_value, renamed_name)
                    elif glsl_type == "sampler3D":
                        assert uniform_value.ndim == 4, (
                            f"Uniform {uniform_name} is not a 4D tensor"
                        )
                        TextureManager.register_texture3d(uniform_value, renamed_name)

            # store this effect's texture later for uniform upload
            self.texture_name_uniform_map.append(effect_textures)

    def upload_uniforms(self, shader_program):
        """No-op for compatibility with viewer. Uniforms are uploaded per-effect in on_frame()"""
        pass

    def _upload_effect_uniforms(
        self, program: int, effect: ImageEffect, effect_index: int
    ):
        """Upload uniforms for a single effect"""
        _, uniforms = effect.get_shader_info()
        effect_textures = self.texture_name_uniform_map[effect_index]
        for uniform_name, uniform_value in uniforms.items():
            # Grab the renamed texture name if it exists
            renamed_uniform_name = effect_textures.get(uniform_name, uniform_name)
            # Re-upload texture data to GPU in case the tensor changed
            if isinstance(uniform_value, torch.Tensor) and uniform_value.ndim >= 3:
                TextureManager.upload_texture_tensor(
                    uniform_value, renamed_uniform_name
                )
            upload_uniform(program, uniform_name, uniform_value, renamed_uniform_name)

    def _render_fullscreen_quad(self):
        """Render a fullscreen quad using the current shader program"""
        gl.glBindVertexArray(self.quad_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

    def _copy_texture_to_fbo(
        self, source_texture_id: int, target_fbo: int, width: int, height: int
    ):
        """Copy a texture to an FBO by rendering with passthrough shader"""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, target_fbo)
        gl.glViewport(0, 0, width, height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Use passthrough shader
        gl.glUseProgram(get_passthrough_shader_program())

        # Bind source texture
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, source_texture_id)
        gl.glUniform1i(
            gl.glGetUniformLocation(get_passthrough_shader_program(), "main_texture"), 0
        )

        # Render
        self._render_fullscreen_quad()

    def prepare_bypass_output(self, texture: str = None):
        """Prepare the bypass display (original image with crop only)."""
        self._render_bypass_output()

    def restore_from_bypass(self, texture: str = None):
        """Restore after bypass. FBOs still hold the processed result — nothing to do."""
        pass

    def _render_bypass_output(self):
        """Render crop-only to the dedicated bypass FBO, or set source dimensions if no crop."""
        # Clean up old bypass FBO
        if self._bypass_fbo is not None:
            self._bypass_fbo.cleanup()
            self._bypass_fbo = None

        # Get source dimensions
        resolution = TextureManager.get_texture_resolution("main_texture")
        source_width, source_height = resolution[1], resolution[0]

        # Find active crop effect and its index
        crop_effect = None
        crop_index = None
        for i, effect in enumerate(self.stack.effects):
            if isinstance(effect, Crop) and effect.toggled and effect.is_active():
                crop_effect = effect
                crop_index = i
                break

        if crop_effect is None:
            # No active crop — use source texture directly (zero cost)
            self._bypass_width = source_width
            self._bypass_height = source_height
            return

        # Calculate crop output dimensions
        delta_w, delta_h = self.effect_dimension_deltas[crop_index]
        self._bypass_width = source_width + delta_w
        self._bypass_height = source_height + delta_h

        # Create dedicated bypass FBO (separate from the main pool)
        self._bypass_fbo = PingPongFBO(
            self._bypass_width, self._bypass_height, "bypass_crop"
        )

        # Render crop shader to bypass FBO
        original_viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)

        gl.glViewport(0, 0, self._bypass_width, self._bypass_height)
        gl.glBindFramebuffer(
            gl.GL_FRAMEBUFFER, self._bypass_fbo.get_write_fbo_id()
        )
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Use the crop shader program
        program = self.effect_programs[crop_index][0]
        gl.glUseProgram(program)

        # Bind source texture as input
        texture_unit = TextureManager.get_texture_unit("main_texture")
        gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.source_texture_id)
        gl.glUniform1i(
            gl.glGetUniformLocation(program, "main_texture"), texture_unit
        )

        # Upload crop uniforms
        self._upload_effect_uniforms(program, crop_effect, crop_index)

        # Render fullscreen quad
        self._render_fullscreen_quad()
        self._bypass_fbo.swap()

        # Restore state
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glUseProgram(0)
        gl.glViewport(
            original_viewport[0],
            original_viewport[1],
            original_viewport[2],
            original_viewport[3],
        )

    def cleanup(self):
        """Clean up processor resources (FBOs, textures, shaders)."""
        if self._bypass_fbo is not None:
            self._bypass_fbo.cleanup()
            self._bypass_fbo = None

        for fbo in self.fbo_pool.values():
            fbo.cleanup()
        self.fbo_pool.clear()

        for effect_textures in self.texture_name_uniform_map:
            for texture_name in effect_textures.values():
                try:
                    TextureManager.delete_texture(texture_name)
                except (ValueError, KeyError):
                    pass
        self.texture_name_uniform_map.clear()

        for programs in self.effect_programs:
            for program in programs:
                if program:
                    gl.glDeleteProgram(program)
        self.effect_programs.clear()
