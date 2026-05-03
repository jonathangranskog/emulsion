import OpenGL.GL as gl


def check_for_errors():
    error = gl.glGetError()
    if error != gl.GL_NO_ERROR:
        print(f"OpenGL error: {error}")
