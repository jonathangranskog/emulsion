from typing import Callable

import glm
import numpy as np
import OpenGL.GL as gl

from src.interface.shader import compile_shader_program
from src.interface.geometry import get_full_screen_quad_vao

# Vertex shader
VERTEX_SHADER_SOURCE = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
out vec2 TexCoord;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
""".strip()


class QuadRenderer:
    def __init__(self, fragment_shader_source: str):
        self.fragment_shader_source = fragment_shader_source
        self.build_shaders(fragment_shader_source)
        self.quad_vao = get_full_screen_quad_vao()

    def build_shaders(self, fragment_shader_source: str):
        # Delete old shader program if it exists
        if hasattr(self, "shader_program") and self.shader_program is not None:
            gl.glDeleteProgram(self.shader_program)
        self.fragment_shader_source = fragment_shader_source
        self.shader_program = compile_shader_program(
            VERTEX_SHADER_SOURCE, self.fragment_shader_source
        )

    def render(
        self,
        texture_id: int,
        projection: np.ndarray,
        view: np.ndarray,
        model: np.ndarray,
        upload_effects_uniforms: Callable[[], None],
    ):
        gl.glUseProgram(self.shader_program)
        projection_loc = gl.glGetUniformLocation(self.shader_program, "projection")
        view_loc = gl.glGetUniformLocation(self.shader_program, "view")
        model_loc = gl.glGetUniformLocation(self.shader_program, "model")

        gl.glUniformMatrix4fv(projection_loc, 1, gl.GL_FALSE, glm.value_ptr(projection))
        gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, glm.value_ptr(model))

        # Bind texture
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glUniform1i(gl.glGetUniformLocation(self.shader_program, "main_texture"), 0)

        # Bind the uniforms here
        if upload_effects_uniforms is not None:
            upload_effects_uniforms(self.shader_program)

        # Draw quad
        gl.glBindVertexArray(self.quad_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        # Unbind texture
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # Clean up
        gl.glUseProgram(0)
