import OpenGL.GL as gl
import numpy as np

FULL_SCREEN_QUAD_VAO = None


def get_full_screen_quad_vao() -> int:
    global FULL_SCREEN_QUAD_VAO
    if FULL_SCREEN_QUAD_VAO is not None:
        return FULL_SCREEN_QUAD_VAO

    vertices = np.array(
        [
            # positions        # texture coords
            [-1.0, -1.0, 0.0, 0.0],
            [1.0, -1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # indices
    indices = np.array(
        [
            0,
            1,
            2,  # first triangle
            2,
            3,
            0,  # second triangle
        ],
        dtype=np.uint32,
    )
    # Create VAO, VBO, EBO
    quad_vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    ebo = gl.glGenBuffers(1)

    gl.glBindVertexArray(quad_vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(
        gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW
    )

    # Position attribute
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, gl.GLvoidp(0))
    gl.glEnableVertexAttribArray(0)

    # Texture coordinate attribute
    gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, gl.GLvoidp(2 * 4))
    gl.glEnableVertexAttribArray(1)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    gl.glBindVertexArray(0)

    FULL_SCREEN_QUAD_VAO = quad_vao
    return FULL_SCREEN_QUAD_VAO
