import OpenGL.GL as gl

def initializeGL(self):
    version = gl.glGetString(gl.GL_VERSION)
    print(f"OpenGL Version: {version}")


print(gl.glGetString(gl.GL_VERSION))