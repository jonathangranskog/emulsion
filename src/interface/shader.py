import OpenGL.GL as gl

# The vertex shader that is used to copy the texture to the FBO
PASSTHROUGH_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
""".strip()

# The shader that is used to copy the texture to the FBO
PASSTHROUGH_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D main_texture;

void main() {
    // Don't flip Y - coordinates are already correct for FBO rendering
    FragColor = texture(main_texture, TexCoord);
}
""".strip()

PASSTHROUGH_SHADER_PROGRAM = None


def get_passthrough_shader_program():
    global PASSTHROUGH_SHADER_PROGRAM
    if PASSTHROUGH_SHADER_PROGRAM is None:
        PASSTHROUGH_SHADER_PROGRAM = compile_shader_program(
            PASSTHROUGH_VERTEX_SHADER, PASSTHROUGH_FRAGMENT_SHADER
        )
    return PASSTHROUGH_SHADER_PROGRAM


def compile_shader_program(vertex_source: str, fragment_source: str) -> int:
    """Compile and link a shader program from source code

    Args:
        vertex_source: GLSL vertex shader source code
        fragment_source: GLSL fragment shader source code

    Returns:
        Shader program ID

    Raises:
        RuntimeError: If shader compilation or linking fails
    """
    # Compile vertex shader
    vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vertex_shader, vertex_source)
    gl.glCompileShader(vertex_shader)

    # Check vertex shader compilation
    success = gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS)
    if not success:
        info_log = gl.glGetShaderInfoLog(vertex_shader)
        raise RuntimeError(f"Vertex shader compilation failed: {info_log.decode()}")

    # Compile fragment shader
    fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(fragment_shader, fragment_source)
    gl.glCompileShader(fragment_shader)

    # Check fragment shader compilation
    success = gl.glGetShaderiv(fragment_shader, gl.GL_COMPILE_STATUS)
    if not success:
        info_log = gl.glGetShaderInfoLog(fragment_shader)
        raise RuntimeError(f"Fragment shader compilation failed: {info_log.decode()}")

    # Link shader program
    shader_program = gl.glCreateProgram()
    gl.glAttachShader(shader_program, vertex_shader)
    gl.glAttachShader(shader_program, fragment_shader)
    gl.glLinkProgram(shader_program)

    # Check linking
    success = gl.glGetProgramiv(shader_program, gl.GL_LINK_STATUS)
    if not success:
        info_log = gl.glGetProgramInfoLog(shader_program)
        raise RuntimeError(f"Shader program linking failed: {info_log.decode()}")

    # Clean up shaders (they're linked into the program now)
    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)

    return shader_program
