import argparse
import imageio
import itertools
import numpy as np
import os
import time
import pickle
import trimesh
import trimesh.proximity
import trimesh.visual

from multiprocessing import Pool
from .obj import ObjHandle
from .utils import getFileList
from tqdm import tqdm

TOLERANCE = 1e-1


def readPointOffset(filepath):
    closest_face_ids = None
    bCoords = None
    d2S_ratios = None
    boundary_indices = None
    tangent_cp2tv = None
    tex_coords = None
    faces = None
    face_tcs = None
    with open(filepath, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith("v "):
                n = int(line[2:])
                closest_face_ids = np.empty(n, dtype=np.int64)
                bCoords = np.empty((n, 3), dtype=np.float64)
                d2S_ratios = np.empty(n, dtype=np.float64)
                boundary_indices = np.empty((n, 2), dtype=np.int64)
                tangent_cp2tv = np.empty((n, 3), dtype=np.float64)
                for i in range(n):
                    line = f.readline().split()
                    closest_face_ids[i] = int(line[0])
                    bCoords[i] = np.array(list(map(float, line[1:4])))
                    d2S_ratios[i] = float(line[4])
                    boundary_indices[i] = np.array(list(map(int, line[5:7])))
                    tangent_cp2tv[i] = np.array(list(map(float, line[7:10])))
            elif line.startswith("vt "):
                n = int(line[3:])
                tex_coords = np.empty((n, 2), dtype=np.float64)
                for i in range(n):
                    line = f.readline().split()
                    tex_coords[i] = np.array(list(map(float, line[0:2])))
            elif line.startswith("f "):
                line = line[2:].split()
                n, c = int(line[0]), int(line[1])
                faces = np.empty((n, c), dtype=np.int64)
                for i in range(n):
                    line = f.readline().split()
                    faces[i] = np.array(list(map(int, line[0:c])))
            elif line.startswith("f/ft "):
                line = line[5:].split()
                n, c = int(line[0]), int(line[1])
                faces = np.empty((n, c), dtype=np.int64)
                face_tcs = np.empty((n, c), dtype=np.int64)
                for i in range(n):
                    line = f.readline().split()
                    faces[i] = np.array(list(map(int, [p.split('/')[0] for p in line]))) - 1
                    face_tcs[i] = np.array(list(map(int, [p.split('/')[1] for p in line]))) - 1
            line = f.readline()

    return closest_face_ids, bCoords, d2S_ratios, boundary_indices, tangent_cp2tv, faces, tex_coords, face_tcs


def writePointOffset(filepath, closest_face_ids, bCoords, d2S_ratios, boundary_indices, tangent_cp2tv, faces, tex_coords=None, face_tcs=None):
    with open(filepath, 'w') as f:
        vn = closest_face_ids.shape[0]
        f.write("v %d\n" % vn)
        for i in range(vn):
            f.write("%d %f %f %f %f " % (closest_face_ids[i], bCoords[i, 0], bCoords[i, 1], bCoords[i, 2], d2S_ratios[i]))
            f.write("%d %d %f %f %f\n" % (boundary_indices[i, 0], boundary_indices[i, 1],
                                          tangent_cp2tv[i, 0], tangent_cp2tv[i, 1], tangent_cp2tv[i, 2]))
        if tex_coords is not None and tex_coords.size != 0:
            f.write("vt %d\n" % tex_coords.shape[0])
            for tcs in tex_coords:
                f.write("%f %f\n" % (tcs[0], tcs[1]))
        fn, cn = faces.shape
        if face_tcs is not None and face_tcs.size != 0:
            f.write("f/ft %d %d\n" % (fn, cn))
            for verts, tcs in zip(faces, face_tcs):
                for i in range(verts.shape[0]):
                    f.write(f"{verts[i] + 1}")
                    f.write(f"/{tcs[i] + 1}")
                    if i != verts.shape[0] - 1:
                        f.write(" ")
                f.write("\n")
        else:
            f.write("f %d %d\n" % (fn, cn))
            for verts, tcs in zip(faces, face_tcs):
                for i in range(verts.shape[0]):
                    f.write(f"{verts[i] + 1}")
                    if i != verts.shape[0] - 1:
                        f.write(" ")
                f.write("\n")


def triangleAreas(x1, x2, x3):
    v0 = x2 - x1
    v1 = x3 - x1
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    denom = d00 * d11 - d01 * d01
    areas = 0.5 * np.sqrt(denom)
    return areas


def barycentricAndArea(x: np.ndarray,
                       x1: np.ndarray,
                       x2: np.ndarray,
                       x3: np.ndarray):
    '''
    Compute barycentric coordinate of X with respect to triagnle X1 X2 X3, and triangle ares of X1 X2 X3.
        :param x: (n x 3) ndarray.
        :param x1: (n x 3) ndarray.
        :param x2: (n x 3) ndarray.
        :param x3: (n x 3) ndarray.
        :return: barycentric coordinate (n x 3) ndarray, triangle areas (n, ) ndarray
    '''
    v0 = x2 - x1
    v1 = x3 - x1
    v2 = x - x1
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    areas = 0.5 * np.sqrt(denom)
    return np.stack([u, v, w], axis=1), areas


def world2tangent(x1, x2, x3, normals, bCoords, world_coords):
    xs = np.stack([x1, x2, x3], axis=1)
    n = xs.shape[0]
    boundary_indices = np.argpartition(bCoords, kth=1, axis=1)[:, 1:]
    rows = np.arange(n)[:, np.newaxis]
    xs = xs[rows, boundary_indices]
    world2tangent_matrix = np.empty((n, 3, 3))
    world2tangent_matrix[:, :, 2] = normals
    world2tangent_matrix[:, :, 1] = xs[:, 0] - xs[:, 1]
    world2tangent_matrix[:, :, 1] /= np.linalg.norm(world2tangent_matrix[:, :, 1], axis=1, keepdims=True)
    world2tangent_matrix[:, :, 0] = np.cross(normals, world2tangent_matrix[:, :, 1])

    tangent_coords = np.sum(world2tangent_matrix * world_coords[..., np.newaxis], axis=1)

    return boundary_indices, tangent_coords


def tangent2world(x1, x2, x3, boundary_indices, normals, tangent_coords):
    xs = np.stack([x1, x2, x3], axis=1)
    n = xs.shape[0]
    rows = np.arange(n)[:, np.newaxis]
    xs = xs[rows, boundary_indices]
    tangent2world_matrix = np.empty((n, 3, 3))
    tangent2world_matrix[:, 2] = normals
    tangent2world_matrix[:, 1] = xs[:, 0] - xs[:, 1]
    tangent2world_matrix[:, 1] /= np.linalg.norm(tangent2world_matrix[:, 1], axis=1, keepdims=True)
    tangent2world_matrix[:, 0] = np.cross(normals, tangent2world_matrix[:, 1])

    world_coords = np.sum(tangent2world_matrix * tangent_coords[..., np.newaxis], axis=1)

    return world_coords


def computeTransfer(source: trimesh.Trimesh, target: trimesh.Trimesh):
    x1 = source.vertices[source.faces[:, 0]]
    x2 = source.vertices[source.faces[:, 1]]
    x3 = source.vertices[source.faces[:, 2]]
    selected_face_indices = np.argwhere(triangleAreas(x1, x2, x3) >= 0.0).squeeze(1)
    selected_vertex_incides = np.argwhere(np.isin(np.arange(len(source.vertices)), source.faces[selected_face_indices])).squeeze(1)
    selected_vertex_inverse = np.empty(len(source.vertices), dtype=np.int64)
    selected_vertex_inverse[selected_vertex_incides] = np.arange(len(selected_vertex_incides))
    selected_faces = selected_vertex_inverse[source.faces[selected_face_indices]]
    source_selected = trimesh.Trimesh(vertices=source.vertices[selected_vertex_incides], faces=selected_faces)

    if len(target.vertices) >= 10000:
        source_query = trimesh.proximity.ProximityQuery(source_selected)
        with Pool() as p:
            chunk_size = min((len(target.vertices) + os.cpu_count() - 1) // os.cpu_count(), 25600)
            target_triangle_x_list = [target.vertices[i:min(i + chunk_size, len(target.vertices))]
                                      for i in range(0, len(target.vertices), chunk_size)]
            result_list = list(
                tqdm(p.imap(query,
                            zip(itertools.repeat(source_query, len(target_triangle_x_list)),
                                target_triangle_x_list),
                            chunksize=1),
                     total=len(target_triangle_x_list),
                     desc="Find closest"))

        closest_points = np.concatenate([r[0] for r in result_list], axis=0)
        distances = np.concatenate([r[1] for r in result_list], axis=0)
        closest_face_ids = np.concatenate([r[2] for r in result_list], axis=0)
    else:
        closest_points, distances, closest_face_ids = trimesh.proximity.closest_point(source_selected, target.vertices)
    closest_face_ids = selected_face_indices[closest_face_ids]
    closest_points_2_target_vertices = target.vertices - closest_points
    normals = source.face_normals[closest_face_ids]

    x1 = source.vertices[source.faces[closest_face_ids, 0]]
    x2 = source.vertices[source.faces[closest_face_ids, 1]]
    x3 = source.vertices[source.faces[closest_face_ids, 2]]
    bCoords, areas = barycentricAndArea(closest_points, x1, x2, x3)
    d2S_ratios = distances / np.sqrt(areas)
    # d2S_ratios = distances

    boundary_indices, tangent_cp2tv = world2tangent(x1, x2, x3, normals, bCoords, closest_points_2_target_vertices)
    tangent_cp2tv /= np.linalg.norm(tangent_cp2tv, axis=1, keepdims=True)

    return closest_face_ids, bCoords, d2S_ratios, boundary_indices, tangent_cp2tv


def applyTransfer(source: trimesh.Trimesh, closest_face_ids, bCoords, d2S_ratios, boundary_indices, tangent_cp2tv):
    closest_face_points = source.vertices[source.faces[closest_face_ids]]
    closest_face_normals = source.face_normals[closest_face_ids]
    vertices = closest_face_points * bCoords[..., np.newaxis]
    vertices = np.sum(vertices, axis=1)
    # vertices = vertices + distances[:, np.newaxis] * closest_face_normals
    x1 = source.vertices[source.faces[closest_face_ids, 0]]
    x2 = source.vertices[source.faces[closest_face_ids, 1]]
    x3 = source.vertices[source.faces[closest_face_ids, 2]]
    areas = triangleAreas(x1, x2, x3)
    distances = np.sqrt(areas) * d2S_ratios
    # distances = d2S_ratios
    world_cp2tv = tangent2world(x1, x2, x3, boundary_indices, closest_face_normals, tangent_cp2tv)
    vertices += distances[:, np.newaxis] * world_cp2tv

    return vertices


def getColor(image: np.ndarray, uvs: np.ndarray):
    H, W, C = image.shape
    points = uvs * np.array([[W, H]])
    mask = np.all(np.logical_and(uvs > 0., uvs < 1.), axis=1)
    result_color = np.zeros((uvs.shape[0], C))
    points = points[mask]
    points[:, 1] = H - points[:, 1]
    lx = np.clip(np.floor(points[:, 0] - 0.5).astype(np.int32), 0, W)
    rx = np.clip(np.ceil(points[:, 0] - 0.5).astype(np.int32), 0, W)
    uy = np.clip(np.floor(points[:, 1] - 0.5).astype(np.int32), 0, H)
    by = np.clip(np.ceil(points[:, 1] - 0.5).astype(np.int32), 0, H)
    xcoeff1 = (rx + 0.5 - points[:, 0])[:, np.newaxis]
    ycoeff1 = (by + 0.5 - points[:, 1])[:, np.newaxis]
    horizontal_color1 = image[uy, lx] * xcoeff1 + image[uy, rx] * (1 - xcoeff1)
    horizontal_color2 = image[by, lx] * xcoeff1 + image[by, rx] * (1 - xcoeff1)
    result_color[mask] = horizontal_color1 * ycoeff1 + horizontal_color2 * (1 - ycoeff1)
    return result_color.astype(np.uint8)


def pointSign(x1, x2, x3):
    return (x1[:, 0] - x3[:, 0]) * (x2[:, 1] - x3[:, 1]) - (x2[:, 0] - x3[:, 0]) * (x1[:, 1] - x3[:, 1])


def pointsInTriangle(x, x1, x2, x3):
    d1 = pointSign(x, x1, x2)
    d2 = pointSign(x, x2, x3)
    d3 = pointSign(x, x3, x1)

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    return np.logical_not(has_neg & has_pos)


def find_uv_face(index_target_vts_H_W):
    i, target_vts, H, W = index_target_vts_H_W
    target_pixel_lu = np.min(target_vts, axis=0) * np.array([W, H])
    target_pixel_lu = np.maximum(np.floor(target_pixel_lu).astype(np.int32), np.array([0, 0]))
    target_pixel_rb = np.max(target_vts, axis=0) * np.array([W, H])
    target_pixel_rb = np.minimum(np.ceil(target_pixel_rb).astype(np.int32), np.array([W, H]))
    px, py = np.meshgrid((np.arange(target_pixel_lu[0], target_pixel_rb[0]) + 0.5) / W,
                         (np.arange(target_pixel_lu[1], target_pixel_rb[1]) + 0.5) / H,
                         indexing='ij')
    xs = np.stack([px, py], axis=2).reshape((-1, 2))
    points_in_triangle_mask = pointsInTriangle(xs,
                                               target_vts[np.newaxis, 0],
                                               target_vts[np.newaxis, 1],
                                               target_vts[np.newaxis, 2])
    valid_xs = xs[points_in_triangle_mask]
    uv_bCoords, _ = barycentricAndArea(valid_xs,
                                       target_vts[np.newaxis, 0],
                                       target_vts[np.newaxis, 1],
                                       target_vts[np.newaxis, 2])
    points_in_triangle_mask = points_in_triangle_mask.reshape(px.shape)
    target_image_rect_slice = slice(target_pixel_lu[0], target_pixel_rb[0]), slice(target_pixel_lu[1],
                                                                                   target_pixel_rb[1])

    return i, uv_bCoords, target_image_rect_slice, points_in_triangle_mask
    # uvCoords_valid_mask[target_image_rect_slice][points_in_triangle_mask] = True
    # uvCoords_face[target_image_rect_slice][points_in_triangle_mask] = i
    # uvCoords_bcoord[target_image_rect_slice][points_in_triangle_mask] = uv_bCoords


def query(source_query_and_target_triangle_x):
    source_query, target_triangle_x = source_query_and_target_triangle_x
    return source_query.on_surface(target_triangle_x)


def computeTextureTransfer(source: trimesh.Trimesh, target: trimesh.Trimesh, H: int, W: int):
    target_vt = target.visual.uv
    uvCoords_valid_mask = np.zeros((H, W), dtype=np.bool8)
    uvCoords_face = np.empty((H, W), dtype=np.int32)
    uvCoords_bcoord = np.empty((H, W, 3))

    with Pool() as p:
        index_target_vts_H_W = zip(range(len(target.faces)),
                                   target_vt[target.faces],
                                   itertools.repeat(H, len(target.faces)),
                                   itertools.repeat(W, len(target.faces)))
        result_list = list(tqdm(p.imap(find_uv_face, index_target_vts_H_W, chunksize=5), total=len(target.faces), desc="Step 1"))
    for i, uv_bCoords, target_image_rect_slice, points_in_triangle_mask in result_list:
        uvCoords_valid_mask[target_image_rect_slice][points_in_triangle_mask] = True
        uvCoords_face[target_image_rect_slice][points_in_triangle_mask] = i
        uvCoords_bcoord[target_image_rect_slice][points_in_triangle_mask] = uv_bCoords
    uvCoords_face = uvCoords_face[uvCoords_valid_mask]
    uvCoords_bcoord = uvCoords_bcoord[uvCoords_valid_mask]

    target_triangle_xs = target.vertices[target.faces[uvCoords_face]]
    target_triangle_x = np.sum(uvCoords_bcoord[..., np.newaxis] * target_triangle_xs, axis=1)
    source_query = trimesh.proximity.ProximityQuery(source)
    with Pool() as p:
        chunk_size = min((len(target_triangle_x) + os.cpu_count() - 1) // os.cpu_count(), 25600)
        target_triangle_x_list = [target_triangle_x[i:min(i + chunk_size, len(target_triangle_x))]
                                  for i in range(0, len(target_triangle_x), chunk_size)]
        result_list = list(
            tqdm(p.imap(query,
                        zip(itertools.repeat(source_query, len(target_triangle_x_list)),
                            target_triangle_x_list),
                        chunksize=1),
                 total=len(target_triangle_x_list),
                 desc="Step 2"))
    src_closest_points = np.concatenate([r[0] for r in result_list], axis=0)
    src_distances = np.concatenate([r[1] for r in result_list], axis=0)
    src_closest_face_ids = np.concatenate([r[2] for r in result_list], axis=0)

    cp2ttx = target_triangle_x - src_closest_points
    cp2ttx = cp2ttx / np.linalg.norm(cp2ttx, axis=1, keepdims=True)
    normals = source.face_normals[src_closest_face_ids]
    valid_ttx_mask = np.abs(np.sum(normals * cp2ttx, axis=1)) >= 1.0 - TOLERANCE
    # valid_ttx_mask = valid_ttx_mask & (src_distances < 0.2)

    src_x1 = source.vertices[source.faces[src_closest_face_ids[valid_ttx_mask], 0]]
    src_x2 = source.vertices[source.faces[src_closest_face_ids[valid_ttx_mask], 1]]
    src_x3 = source.vertices[source.faces[src_closest_face_ids[valid_ttx_mask], 2]]
    src_x = src_closest_points[valid_ttx_mask]
    src_bCoords, _ = barycentricAndArea(src_x, src_x1, src_x2, src_x3)
    src_uvs = source.visual.uv[source.faces[src_closest_face_ids[valid_ttx_mask]]]
    src_uvCoords = np.sum(src_uvs * src_bCoords[..., np.newaxis], axis=1)
    result_mask = uvCoords_valid_mask.copy()
    result_mask[result_mask] = valid_ttx_mask
    # c = np.sum(result_mask)
    texture_args = {
        "width": W,
        "height": H,
        "mask": result_mask,
        "uv_coords": src_uvCoords
    }

    return texture_args


def applyTextureTransfer(height, width, mask, uv_coords, texture: np.ndarray):
    H = height
    W = width
    result_mask = mask
    src_uvCoords = uv_coords

    result = np.zeros((H, W, 3), dtype=np.uint8)
    src_colors = getColor(texture, src_uvCoords)
    result[result_mask] = src_colors

    result = result.transpose((1, 0, 2))[::-1, :, :]
    return result


def textureTransfer(source: trimesh.Trimesh, target: trimesh.Trimesh, texture: np.ndarray):
    H, W, _ = texture.shape
    H //= 2
    W //= 2
    # uCoords, vCoords = np.meshgrid(np.linspace(0.5, H-0.5, H), np.linspace(0.5, W-0.5,W))
    # uvCoords = np.stack([uCoords, vCoords], axis=2)
    # uvCoords = uvCoords.reshape((H * W, 2))
    target_vt = target.visual.uv
    uvCoords_valid_mask = np.zeros((H, W), dtype=np.bool8)
    uvCoords_face = np.empty((H, W), dtype=np.int32)
    uvCoords_bcoord = np.empty((H, W, 3))

    # pbar = tqdm(range(len(target.faces)))
    # pbar.set_description("Step 1")
    # for i in pbar:
    #     find_uv_face((i, target_vt[target.faces[i]]))
    # uvCoords = uvCoords[uvCoords_valid_mask]
    with Pool() as p:
        index_target_vts_H_W = zip(range(len(target.faces)),
                                   target_vt[target.faces],
                                   itertools.repeat(H, len(target.faces)),
                                   itertools.repeat(W, len(target.faces)))
        result_list = list(tqdm(p.imap(find_uv_face, index_target_vts_H_W, chunksize=5), total=len(target.faces), desc="Step 1"))
    for i, uv_bCoords, target_image_rect_slice, points_in_triangle_mask in result_list:
        uvCoords_valid_mask[target_image_rect_slice][points_in_triangle_mask] = True
        uvCoords_face[target_image_rect_slice][points_in_triangle_mask] = i
        uvCoords_bcoord[target_image_rect_slice][points_in_triangle_mask] = uv_bCoords
    uvCoords_face = uvCoords_face[uvCoords_valid_mask]
    uvCoords_bcoord = uvCoords_bcoord[uvCoords_valid_mask]

    target_triangle_xs = target.vertices[target.faces[uvCoords_face]]
    target_triangle_x = np.sum(uvCoords_bcoord[..., np.newaxis] * target_triangle_xs, axis=1)
    source_query = trimesh.proximity.ProximityQuery(source)
    with Pool() as p:
        chunk_size = min((len(target_triangle_x) + os.cpu_count() - 1) // os.cpu_count(), 25600)
        target_triangle_x_list = [target_triangle_x[i:min(i + chunk_size, len(target_triangle_x))]
                                  for i in range(0, len(target_triangle_x), chunk_size)]
        result_list = list(
            tqdm(p.imap(query,
                        zip(itertools.repeat(source_query, len(target_triangle_x_list)),
                            target_triangle_x_list),
                        chunksize=1),
                 total=len(target_triangle_x_list),
                 desc="Step 2"))
    src_closest_points = np.concatenate([r[0] for r in result_list], axis=0)
    src_distances = np.concatenate([r[1] for r in result_list], axis=0)
    src_closest_face_ids = np.concatenate([r[2] for r in result_list], axis=0)
    # start_time = time.time()
    # src_closest_points, src_distances, src_closest_face_ids = trimesh.proximity.closest_point(source, target_triangle_x)
    # end_time = time.time()
    # t = end_time - start_time
    cp2ttx = target_triangle_x - src_closest_points
    cp2ttx = cp2ttx / np.linalg.norm(cp2ttx, axis=1, keepdims=True)
    normals = source.face_normals[src_closest_face_ids]
    valid_ttx_mask = np.abs(np.sum(normals * cp2ttx, axis=1)) >= 1.0 - TOLERANCE

    src_x1 = source.vertices[source.faces[src_closest_face_ids[valid_ttx_mask], 0]]
    src_x2 = source.vertices[source.faces[src_closest_face_ids[valid_ttx_mask], 1]]
    src_x3 = source.vertices[source.faces[src_closest_face_ids[valid_ttx_mask], 2]]
    src_x = src_closest_points[valid_ttx_mask]
    src_bCoords, _ = barycentricAndArea(src_x, src_x1, src_x2, src_x3)
    src_uvs = source.visual.uv[source.faces[src_closest_face_ids[valid_ttx_mask]]]
    src_uvCoords = np.sum(src_uvs * src_bCoords[..., np.newaxis], axis=1)
    src_colors = getColor(texture, src_uvCoords)
    result = np.zeros((H, W, 3), dtype=np.uint8)
    result_mask = uvCoords_valid_mask.copy()
    result_mask[result_mask] = valid_ttx_mask
    result[result_mask] = src_colors
    # result = result.reshape((H, W, 3))
    result = result.transpose((1, 0, 2))[::-1, :, :]
    return result

# def getFileList(root, ext, recursive=False):
#     files = []
#     dirs = os.listdir(root)
#     while len(dirs) > 0:
#         path = dirs.pop()
#         fullname = os.path.join(root, path)
#         if os.path.isfile(fullname) and fullname.endswith(ext):
#             files.append(path)
#         elif recursive and os.path.isdir(fullname):
#             for s in os.listdir(fullname):
#                 newDir = os.path.join(path, s)
#                 dirs.append(newDir)
#     files = sorted(files)
#     return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Nu to USC mesh topology transfer.')
    parser.add_argument("--record", action="store_true", help="RECORD MODE which record source to target mapping to conf.")
    parser.add_argument("--recursive", action="store_true",
                        help="Transfer all meshes in source directory recursively. Valid in transfer mode ONLY.")
    parser.add_argument("--source", type=str, default="data/nu.obj", help="Mesh or directory of meshes with original topology.")
    parser.add_argument("--target", type=str, default="data/usc.obj", help="Mesh or directory of meshes with target topology.")
    parser.add_argument("--texture", type=str, default=None, help="texture transfer args path.")
    parser.add_argument("--res", type=int, default=2048, help="texture resolution.")
    parser.add_argument("--map", type=str, default="data/nu2usc.map", help="File that record source to target mapping.")
    args = parser.parse_args()

    if args.record:
        source_handle = ObjHandle(args.source, triangulate=True)
        target_handle = ObjHandle(args.target, triangulate=True)
        source = trimesh.Trimesh(vertices=source_handle.vertices, faces=source_handle.faces, process=False)
        # print(type(target_handle.faces))
        # print(target_handle.faces)
        for i in target_handle.faces:
            assert len(i) == 3, i
        target = trimesh.Trimesh(vertices=target_handle.vertices, faces=target_handle.faces, process=False)
        closest_face_ids, bCoords, d2S_ratios, boundary_indices, tangent_cp2tv = computeTransfer(source, target)
        writePointOffset(args.map, closest_face_ids, bCoords, d2S_ratios, boundary_indices, tangent_cp2tv,
                         target_handle.faces, target_handle.texcoords, target_handle.face_texs)
        if args.texture is not None:
            source = trimesh.load(args.source, process=False)
            target = trimesh.load(args.target, process=False)
            texture_args = computeTextureTransfer(source, target, args.res, args.res)
            pickle.dump(texture_args, open(args.texture, 'wb'))
    else:
        closest_face_ids, bCoords, d2S_ratios, boundary_indices, tangent_cp2tv, faces, tex_coords, face_tcs = readPointOffset(args.map)
        if args.texture is not None:
            texture_args = pickle.load(open(args.texture, 'rb'))
        if os.path.isdir(args.source):
            sources = getFileList(args.source, '.obj', recursive=args.recursive)
            os.makedirs(args.target, exist_ok=True)
            for source_file in sources:
                basename, ext = os.path.splitext(source_file)
                print(f"Transfer {source_file}")
                source_handle = ObjHandle(os.path.join(args.source, source_file), triangulate=True)
                source = trimesh.Trimesh(vertices=source_handle.vertices, faces=source_handle.faces, process=False)
                target_verts = applyTransfer(source, closest_face_ids, bCoords, d2S_ratios, boundary_indices, tangent_cp2tv)
                target_handle = ObjHandle(vertices=target_verts, texcoords=tex_coords, faces=faces, face_texs=face_tcs,
                                          mtl=source_handle.mtl, obj_material=source_handle.obj_material, triangulate=True)
                dir, _ = os.path.split(source_file)
                os.makedirs(os.path.join(args.target, dir), exist_ok=True)
                target_file = os.path.join(args.target, basename + ext)
                target_handle.write(target_file)

                possible_tex_file = os.path.join(args.source, basename + '.jpg')
                if args.texture is not None and source_handle.texcoords.size != 0 and tex_coords is not None\
                        and os.path.exists(possible_tex_file) and os.path.isfile(possible_tex_file):
                    print("With texture")
                    texture = imageio.imread(os.path.join(args.source, basename + '.jpg'))
                    # source = trimesh.load(os.path.join(args.source, source_file), process=False)
                    # target = trimesh.load(target_file, process=False)
                    new_texture = applyTextureTransfer(texture=texture, **texture_args)
                    # new_texture = textureTransfer(source, target, texture)
                    imageio.imwrite(os.path.join(args.target, basename + '.jpg'), new_texture, jpg_quality=95)
        else:
            source_handle = ObjHandle(args.source, triangulate=True)
            source = trimesh.Trimesh(vertices=source_handle.vertices, faces=source_handle.faces, process=False)
            target_verts = applyTransfer(source, closest_face_ids, bCoords, d2S_ratios, boundary_indices, tangent_cp2tv)
            target_handle = ObjHandle(vertices=target_verts, texcoords=tex_coords, faces=faces, face_texs=face_tcs,
                                      mtl=source_handle.mtl, obj_material=source_handle.obj_material, triangulate=True)
            target_handle.write(args.target)

            basename, ext = os.path.splitext(args.source)
            if args.texture and source_handle.texcoords.size != 0 \
                    and os.path.exists(basename + '.jpg') and os.path.isfile(basename + '.jpg'):
                texture = imageio.imread(basename + '.jpg')
                # source = trimesh.load(args.source, process=False)
                # target = trimesh.load(args.target, process=False)
                # start_time = time.time()
                new_texture = applyTextureTransfer(texture=texture, **texture_args)
                # end_time = time.time()
                # interval = end_time - start_time
                # print(f"It take {interval} seconds")
                imageio.imwrite(basename + '.jpg', new_texture, jpg_quality=95)
