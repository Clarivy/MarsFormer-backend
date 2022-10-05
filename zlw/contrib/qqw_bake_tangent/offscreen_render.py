import OpenGL.GL as gl
import glfw
import sys
import numpy as np


class GLFW_OFFSCREEN_RENDER:
    def __init__(self, resolution):
        # Initialize the library
        if not glfw.init():
            sys.exit()

        # Off-screen contexts
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        offscreen_context = glfw.create_window(resolution[0], resolution[1], "", None, None)

        glfw.make_context_current(offscreen_context)

        # Build frame renderbuffer
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

        # Build frame color buffer
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB32F, resolution[0], resolution[1], 0, gl.GL_RGB, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glViewport(0, 0, resolution[0], resolution[1])
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tex, 0)

        # Build frame depth buffer
        depth = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, depth)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, resolution[0], resolution[1], 0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
        gl.glViewport(0, 0, resolution[0], resolution[1])
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, depth, 0)

        if(gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE):
            print("ERROR::FRAMEBUFFER:: Framebuffer is not complete!")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        self.fbo = fbo
        self.tex = tex
        self.depth = depth
        self.resolution = resolution
        self.offscreen_context = offscreen_context

        # gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo) is need before draw
    # def __del__(self):
    #     # body of destructor
    #     gl.glDeleteTextures(1, [self.tex])
    #     gl.glDeleteTextures(1, [self.depth])
    #     gl.glDeleteFramebuffers(1, [self.fbo])
    #     print("destroying")
    #     glfw.destroy_window(self.offscreen_context)
    #     print("destroyed")

    def save_img(self):
        # save image
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        image_buffer = gl.glReadPixels(0, 0, self.resolution[0], self.resolution[1], gl.GL_RGB, gl.GL_FLOAT)
        image = np.frombuffer(image_buffer, dtype=np.float32).reshape(self.resolution[1], self.resolution[0], 3)
        image = image[::-1]  # read from bottom left
        return image
