import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import cv2
import numpy as np
import sys
import ctypes
import time
import zlw
import tqdm
from .offscreen_render import *
import zlw_cpp


VERTEX_SHADER = """
# version 330

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec2 a_uv;

out vec2 v_uv;
out vec3 v_position;

uniform mat4 trans_inv;
uniform mat4 extrinsic;
uniform mat4 intrinsic;

void main() {
    gl_Position = intrinsic*extrinsic*trans_inv*vec4(a_position,1);
    v_position=vec3(gl_Position)/gl_Position.w;
    v_uv = a_uv;
}
"""

FRAGMENT_SHADER = """
# version 330
in vec2 v_uv;
in vec3 v_tangent;
in vec3 v_position;

out vec4 frag_color;
uniform sampler2D texture_texture_input;

void main() {
    vec3 texture_input =vec3(texture(texture_texture_input, v_uv));

    frag_color = vec4(texture_input.x, texture_input.y, texture_input.z, 1.0);
}
"""


def render_mesh(texture_input, mesh, resolution, intrinsic3x4, extrinsic: "4x4", affine: "4x4" = np.identity(4)):

    intrinsic = np.zeros((4, 4))

    d = intrinsic3x4[:2, 2] / np.array(resolution) * 2 - 1
    k = (d + 1) * intrinsic3x4[[0, 1], [0, 1]] / intrinsic3x4[:2, 2]

    intrinsic[[0, 1], [0, 1]] = k
    intrinsic[:2, 2] = d
    intrinsic[2, 2] = 1
    intrinsic[2, 3] = -0.01
    intrinsic[3, 2] = 1

    """
              / +Z
             /
    OPENCV  +------> +X
    (CAM)   |
            |
            v +Y


            ^ +Y
    NDC     |
            |
            +------> +X
           /
          / +Z

    """

    intrinsic = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])@intrinsic

    last_resolution, renderer = getattr(render_mesh, "last_resolution", (None, None))
    if resolution != last_resolution:
        if renderer is not None:
            del renderer
        renderer = GLFW_OFFSCREEN_RENDER(resolution)
        setattr(render_mesh, "last_resolution", (resolution, renderer))

    last_sign = getattr(render_mesh, "last_sign", None)
    last_data = getattr(render_mesh, "last_data", {})
    if mesh == last_sign:
        shader = last_data["shader"]
        indices = last_data["indices"]

    else:

        obj = zlw_cpp.read_obj(mesh)

        vertex_attribute, indices = zlw_cpp.build_vertex_attribute(obj.fvs, obj.fvts, obj.vs, obj.vts)

        if "VBO" in last_data:
            glDeleteBuffers(1, [last_data["VBO"]])
        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertex_attribute.itemsize * len(vertex_attribute), vertex_attribute, GL_DYNAMIC_DRAW)

        if "EBO" in last_data:
            glDeleteBuffers(1, [last_data["EBO"]])
        EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.itemsize * len(indices), indices, GL_STATIC_DRAW)

        # get the position from vertex shader
        step = 4 * 5

        a_position = 0
        glVertexAttribPointer(a_position, 3, GL_FLOAT, GL_FALSE, step, ctypes.c_void_p(0 * 4))
        glEnableVertexAttribArray(a_position)

        a_uv = 1
        glVertexAttribPointer(a_uv, 2, GL_FLOAT, GL_FALSE, step, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(a_uv)

        height, width = texture_input.shape[:2]

        if "TEX" in last_data:
            glDeleteTextures(1, [last_data["TEX"]])
        TEX = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0 + 0)
        glBindTexture(GL_TEXTURE_2D, TEX)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, texture_input)
        glEnable(GL_TEXTURE_2D)

        shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
        glUseProgram(shader)
        glUniform1i(glGetUniformLocation(shader, "texture_texture_input"), 0)

        last_data = {}
        last_data["shader"] = shader
        last_data["indices"] = indices
        last_data["VBO"] = VBO
        last_data["EBO"] = EBO
        last_data["TEX"] = TEX
        setattr(render_mesh, "last_data", last_data)
        setattr(render_mesh, "last_sign", mesh)

    glUniformMatrix4fv(glGetUniformLocation(shader, "trans_inv"), 1, GL_TRUE, affine)
    glUniformMatrix4fv(glGetUniformLocation(shader, "extrinsic"), 1, GL_TRUE, extrinsic)
    glUniformMatrix4fv(glGetUniformLocation(shader, "intrinsic"), 1, GL_TRUE, intrinsic)

    glBindFramebuffer(GL_FRAMEBUFFER, renderer.fbo)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)

    glClearColor(0.0, 0.0, 0.0, 1.0)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDrawElements(GL_TRIANGLES, indices.shape[0], GL_UNSIGNED_INT, None)

    img = renderer.save_img()
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return img


if __name__ == "__main__":

    # texture_input = cv2.imread(r"Y:\ProjectHH_Delivery\003\3_Texture_Package\process\00\UV_diffuse_merged.png")[::-1, :, ::-1].astype(np.float32) / 255
    # mesh_path = r"Y:\ProjectHH_Delivery\003\3_Texture_Package\inter\00\model\translated.obj"
    # resolution = (4096, 2304)

    # intrinsic3x4 = np.loadtxt(r"Y:\ProjectHH_Delivery\003\3_Texture_Package\para\intrinsic.txt")[:3]
    # extrinsic = np.loadtxt(r"Y:\ProjectHH_Delivery\003\3_Texture_Package\para\extrinsic.txt")[:4]
    # translate = np.loadtxt(r"Y:\ProjectHH_Delivery\003\3_Texture_Package\para\translate.txt")
    # affine = np.linalg.inv(translate)

    texture_input = cv2.imread(r"E:\20220817_zqx_4d\20220817_zqx\obj_template\2_000001.jpg")[::-1, :, ::-1].astype(np.float32) / 255
    mesh_path = r"E:\20220817_zqx_4d\20220817_zqx\obj_template\2_000000.obj"

    resolution = (256, 256)

    f = 3000

    intrinsic = np.array([
        [f, 0, resolution[0] / 2 - 0.5],
        [0, f, resolution[1] / 2 - 0.5],
        [0, 0, 1],
    ])

    extrinsic = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 100],
        [0, 0, 0, 1],
    ])

    image = render_mesh(texture_input, mesh_path, resolution, intrinsic, extrinsic)

    cv2.imwrite("rendered.png", image[:, :, ::-1] * 255)
