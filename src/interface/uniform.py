from typing import Any

import numpy as np
import torch
import OpenGL.GL as gl
from src.interface.texture import TextureManager


def infer_glsl_type(value: Any) -> str:
    """Infer GLSL uniform type from Python value"""
    if isinstance(value, (int, np.integer)):
        return "int"
    elif isinstance(value, (float, np.floating)):
        return "float"
    elif isinstance(value, bool):
        return "bool"
    elif isinstance(value, (list, tuple)):
        length = len(value)
        if length == 2:
            return "vec2"
        elif length == 3:
            return "vec3"
        elif length == 4:
            return "vec4"
        else:
            raise ValueError(f"Unsupported vector length: {length}")
    elif isinstance(value, (torch.Tensor, np.ndarray)):
        # For textures - we'll handle these specially
        if value.ndim == 4:
            return "sampler3D"  # 3D LUT
        elif value.ndim == 3:
            return "sampler2D"  # 2D texture/gradient
        elif value.ndim == 1 and value.shape[0] == 3:
            return "vec3"
        elif value.ndim == 1 and value.shape[0] == 4:
            return "vec4"
        else:
            raise ValueError(f"Unsupported tensor dimensions: {value.ndim}")
    else:
        raise ValueError(f"Unsupported uniform type: {type(value)}")


def upload_uniform(
    shader_program: int,
    uniform_name: str,
    uniform_value: Any,
    texture_name: str = None,
):
    # Get the uniform location from the shader program
    location = gl.glGetUniformLocation(shader_program, uniform_name)
    if location == -1:
        print(f"Warning: Uniform {uniform_name} not found in shader")
        return

    # Get the type and upload the value
    glsl_type = infer_glsl_type(uniform_value)
    match glsl_type:
        case "float":
            gl.glUniform1f(location, float(uniform_value))
        case "int":
            gl.glUniform1i(location, int(uniform_value))
        case "vec2":
            gl.glUniform2f(location, float(uniform_value[0]), float(uniform_value[1]))
        case "vec3":
            gl.glUniform3f(
                location,
                float(uniform_value[0]),
                float(uniform_value[1]),
                float(uniform_value[2]),
            )
        case "vec4":
            gl.glUniform4f(
                location,
                float(uniform_value[0]),
                float(uniform_value[1]),
                float(uniform_value[2]),
                float(uniform_value[3]),
            )
        case "sampler2D":
            texture_unit = TextureManager.get_texture_unit(texture_name or uniform_name)
            texture_id = TextureManager.get_texture_id(texture_name or uniform_name)
            gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
            gl.glUniform1i(location, texture_unit)
        case "sampler3D":
            texture_unit = TextureManager.get_texture_unit(texture_name or uniform_name)
            texture_id = TextureManager.get_texture_id(texture_name or uniform_name)
            gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
            gl.glBindTexture(gl.GL_TEXTURE_3D, texture_id)
            gl.glUniform1i(location, texture_unit)
