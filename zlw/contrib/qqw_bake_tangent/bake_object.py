import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import cv2
import numpy as np
import sys
from .read_obj import read_obj
import ctypes
import time
import zlw
import tqdm
from .offscreen_render import *


def compute_TB(mesh):
    # mesh: v (Nv,3) vt (Nuv,2) f (Nf,3,2)
    # face_pos (Nf,3,3) face_uv (Nf,3,2)
    face_pos = mesh.v[mesh.f[:, :, 0].reshape((-1,))].reshape((-1, 3, 3))
    face_uv = mesh.vt[mesh.f[:, :, 1].reshape((-1,))].reshape((-1, 3, 2))
    dpos1 = face_pos[:, 1, :] - face_pos[:, 0, :]
    dpos2 = face_pos[:, 2, :] - face_pos[:, 0, :]
    duv1 = face_uv[:, 1, :] - face_uv[:, 0, :]
    duv2 = face_uv[:, 2, :] - face_uv[:, 0, :]
    T_face = (dpos1 * duv2[:, 1:2] - dpos2 * duv1[:, 1:2]) / (duv1[:, 0:1] * duv2[:, 1:2] - duv2[:, 0:1] * duv1[:, 1:2])
    T_face /= np.linalg.norm(T_face, axis=-1)[:, None]
    N_face = np.cross(face_pos[:, 0, :] - face_pos[:, 1, :], face_pos[:, 1, :] - face_pos[:, 2, :])
    N_face /= np.linalg.norm(N_face, axis=-1)[:, None]

    vertex_t = np.zeros_like(mesh.v)
    vertex_n = np.zeros_like(mesh.v)
    for i in tqdm.tqdm(range(len(mesh.f))):
        vertex_t[mesh.f[i, :, 0]] += T_face[i]
        vertex_n[mesh.f[i, :, 0]] += N_face[i]
    T_vertex = vertex_t / np.linalg.norm(vertex_t, axis=-1)[:, None]
    N_vertex = vertex_n / np.linalg.norm(vertex_n, axis=-1)[:, None]
    return T_vertex, N_vertex


def bake_normal(normal_input, mesh, resolution, normal_model):
    T_vertex, N_vertex = compute_TB(mesh)
    T_uv = np.zeros((mesh.vt.shape[0], 3))
    N_uv = np.zeros((mesh.vt.shape[0], 3))
    T_uv[mesh.f[:, :, 1].reshape(-1)] = T_vertex[mesh.f[:, :, 0].reshape(-1)]
    N_uv[mesh.f[:, :, 1].reshape(-1)] = N_vertex[mesh.f[:, :, 0].reshape(-1)]

    DISPLAY_HEIGHT, DISPLAY_WIDTH = resolution

    renderer = GLFW_OFFSCREEN_RENDER(resolution)

    vertex_attribute = np.zeros((len(mesh.vt), 11), np.float32)
    vertex_attribute[:, :2] = mesh.vt * 2 - 1
    vertex_attribute[:, 3:5] = mesh.vt
    vertex_attribute[:, 5:8] = T_uv
    vertex_attribute[:, 8:11] = N_uv

    indices = mesh.f[:, :, 1].reshape(-1)

    VERTEX_SHADER = """
    # version 330

    layout (location = 0) in vec3 a_position;
    layout (location = 1) in vec2 a_uv;
    layout (location = 2) in vec3 a_tangent;
    layout (location = 3) in vec3 a_normal;

    out vec2 v_uv;
    out vec3 v_tangent;
    out vec3 v_normal;


    void main() {
        gl_Position = vec4(a_position.x,a_position.y,0,1);
        v_uv = a_uv;
        v_tangent = a_tangent;
        v_normal = a_normal;
    }
    """

    FRAGMENT_SHADER = """
    # version 330
    in vec2 v_uv;
    in vec3 v_tangent;
    in vec3 v_normal;
    out vec4 frag_color;
    uniform sampler2D texture_normal_input;
    uniform sampler2D texture_normal_model;

    void main() {
        vec3 normal_input = normalize(vec3(texture(texture_normal_input, v_uv)));
        vec3 normal_model = normalize(vec3(texture(texture_normal_model, v_uv)));

        vec3 bitangent = normalize(cross(normal_model, v_tangent));
        vec3 tangent = cross(bitangent, normal_model);

        mat3 tbn = mat3(tangent, bitangent, normal_model);

        vec3 normal_tangent = normal_input*tbn;
        //frag_color = vec4(normal_model.x, normal_model.y, normal_model.z, 1.0);
        frag_color = vec4(normal_tangent.x, normal_tangent.y, normal_tangent.z, 1.0);
    }
    """

    # Compile The Program and shaders

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertex_attribute.itemsize * vertex_attribute.shape[0] * vertex_attribute.shape[1], vertex_attribute.reshape(-1), GL_DYNAMIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.itemsize * len(indices), indices, GL_STATIC_DRAW)

    # get the position from vertex shader
    step = 4 * 11

    a_position = 0
    glVertexAttribPointer(a_position, 3, GL_FLOAT, GL_FALSE, step, ctypes.c_void_p(0 * 4))
    glEnableVertexAttribArray(a_position)

    a_uv = 1
    glVertexAttribPointer(a_uv, 2, GL_FLOAT, GL_FALSE, step, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(a_uv)

    tangent = 2
    glVertexAttribPointer(tangent, 3, GL_FLOAT, GL_FALSE, step, ctypes.c_void_p(5 * 4))
    glEnableVertexAttribArray(tangent)

    normal = 3
    glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, step, ctypes.c_void_p(8 * 4))
    # glEnableVertexAttribArray(normal)

    height, width = normal_input.shape[:2]

    TEX = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0 + 0)
    glBindTexture(GL_TEXTURE_2D, TEX)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, normal_input)
    glEnable(GL_TEXTURE_2D)

    TEX2 = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0 + 1)
    glBindTexture(GL_TEXTURE_2D, TEX2)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, normal_model)
    glEnable(GL_TEXTURE_2D)

    glUseProgram(shader)
    glUniform1i(glGetUniformLocation(shader, "texture_normal_input"), 0)
    glUniform1i(glGetUniformLocation(shader, "texture_normal_model"), 1)

    glEnable(GL_DEPTH_TEST)

    glBindFramebuffer(GL_FRAMEBUFFER, renderer.fbo)
    glClearColor(0, 0, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDrawElements(GL_TRIANGLES, indices.shape[0], GL_UNSIGNED_INT, None)
    img = renderer.save_img()
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    return img


def main(mesh_path, normal_input_path, normal_model_path, output_path):
    mesh = read_obj(mesh_path)
    normal_input = zlw.read_normal(normal_input_path, use_xyz=True)[::-1]
    normal_model = zlw.read_normal(normal_model_path, use_xyz=True)[::-1]
    start = time.time()
    normal_baked = bake_normal(normal_input, mesh, (normal_model.shape[1], normal_model.shape[0]), normal_model)
    print(f"baked after {time.time() - start:.3f} s")
    zlw.save_normal(output_path, normal_baked, use_xyz=True)


if __name__ == "__main__":
    main(
        r"Y:\ProjectHH_Delivery\001\3_Texture_Package\inter\00\model\translated.obj",
        r"Y:\ProjectHH_Delivery\001\3_Texture_Package\process\00\total_matrix.png",
        r"Y:\ProjectHH_Delivery\001\3_Texture_Package\process\00\UV_to_normal_u16.png",
        "tangent_normal.png",
    )
