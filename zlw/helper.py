import argparse
import re
import numpy as np
import xml.dom.minidom
import os
import glob
import cv2
import numpy as np
from multiprocessing.pool import ThreadPool
import functools
import shutil
import subprocess
import tqdm
import pickle
import sys
import warnings
import scipy.optimize as opt
import time
sys.path = [os.path.dirname(__file__)] + sys.path


NAMES_USC55 = ["browDown_L", "browDown_R", "browInnerUp_L", "browInnerUp_R", "browOuterUp_L", "browOuterUp_R", "cheekPuff_L", "cheekPuff_R", "cheekRaiser_L", "cheekRaiser_R", "cheekSquint_L", "cheekSquint_R", "eyeBlink_L", "eyeBlink_R", "eyeLookDown_L", "eyeLookDown_R", "eyeLookIn_L", "eyeLookIn_R", "eyeLookOut_L", "eyeLookOut_R", "eyeLookUp_L", "eyeLookUp_R", "eyeSquint_L", "eyeSquint_R", "eyeWide_L", "eyeWide_R", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimple_L", "mouthDimple_R", "mouthFrown_L", "mouthFrown_R", "mouthFunnel", "mouthLeft", "mouthLowerDown_L", "mouthLowerDown_R", "mouthPress_L", "mouthPress_R", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmile_L", "mouthSmile_R", "mouthStretch_L", "mouthStretch_R", "mouthUpperUp_L", "mouthUpperUp_R", "noseSneer_L", "noseSneer_R"]


class Timer:
    def __init__(self, prompt="time"):
        self.start_time = 0
        self.prompt = prompt

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        print(f"{self.prompt} : {time.time() - self.start_time} s")


def dive_path(func):
    def wrapper(path, *args, **kwargs):
        print(f"current path is {path}\n", end="")
        ret = func(path, *args, **kwargs)
        return ret
    return wrapper


def imencodewrite(path, img):
    try:
        suffix = "." + path.split(".")[-1]
        binary = cv2.imencode(suffix, img)[1].tobytes()
    except Exception as e:
        print("failed to encode", suffix)
        raise e
    open(path, "wb").write(binary)


cv2.imwrite = imencodewrite


def read_normal(path, small=1, use_xyz=False):
    if not use_xyz:
        raise Exception("use_xyz=False")
    x = cv2.imread(path, -1)
    x = x[..., :3]
    t = x.dtype
    x = x[::small, ::small, ::-1]
    x = x.astype(np.float32)
    x[np.linalg.norm(x, ord=np.inf, axis=2) == 0] = np.nan
    if t == np.uint16:
        x = (x - 32768) / 32767
    elif t == np.uint8:
        x = (x - 128) / 127
    x[np.linalg.norm(x, ord=np.inf, axis=2) == 0] = np.nan
    return x


def save_normal(path, n, use_xyz=False, small=1):
    if not use_xyz:
        raise Exception("use_xyz=False")
    cv2.imwrite(path, (n[..., ::-1] * 32767 + 32768).clip(0, 65535).astype(np.uint16))


def seamless_blur(img, ksize=111, use_iter=False):

    def blur_iter(x, ksize, sigma):
        ksize_real = 31
        """
        n=4**2
        """
        # sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        n = (ksize[0] / ksize_real)**2
        for i in range(int(n)):
            x = cv2.GaussianBlur(x, (ksize_real, ksize_real), 0)
        return x
    blur = blur_iter if use_iter else cv2.GaussianBlur

    img = img.copy()
    white = np.ones(img.shape[:2])
    if len(img.shape) == 3:
        outer = np.isnan(img[..., 0])
    else:
        outer = np.isnan(img)
    white[outer] = 0
    img[outer] = 0
    if len(img.shape) == 3:
        fm = blur(white, (ksize, ksize), 0)[..., None]
    else:
        fm = blur(white, (ksize, ksize), 0)
    blured = blur(img, (ksize, ksize), 0) / fm
    blured[outer] = np.nan
    return blured


def normalize(normal):
    return normal / np.linalg.norm(normal, axis=2)[..., None]


class Obj:
    vs = None
    vts = None
    fvs = None
    fvts = None

    def tri(self):
        self.fvs = np.concatenate([self.fvs[:, [0, 1, 2]], self.fvs[:, [0, 2, 3]]])
        self.fvts = np.concatenate([self.fvts[:, [0, 1, 2]], self.fvts[:, [0, 2, 3]]])

    def write(self, path):
        f = open(path, "w")
        f.write("\n".join(
            [f"v {i[0]:.6f} {i[1]:.6f} {i[2]:.6f}" for i in self.vs] +
            [f"vt {i[0]:.6f} {i[1]:.6f}" for i in self.vts] +
            [(f"f " + " ".join([f"{ii+1}/{jj+1}" for ii, jj in zip(i, j)])) for i, j in zip(self.fvs, self.fvts)]
        ))


def read_obj(obj_path, only_vs=False):
    objfile = open(obj_path, encoding="utf8").read().strip().split("\n")
    if only_vs:
        obj = Obj()
        obj.vs = np.array([[float(j) for j in i[2:].strip().split()] for i in filter(lambda x:x.startswith("v "), objfile)], np.float32)
        return obj

    vs = []
    vts = []
    fvs = []
    fvts = []
    for line in objfile:
        if line.startswith("v "):
            vs.append(list(map(float, line[2:].strip().split())))
        elif line.startswith("vt "):
            vts.append(list(map(float, line[2:].strip().split())))
        elif line.startswith("f "):
            fv = []
            fvt = []
            for i in line[2:].strip().split():
                vth, vtth = i.split("/")[:2]
                vth = int(vth) - 1
                vtth = int(vtth) - 1
                fv.append(vth)
                fvt.append(vtth)

            fvs.append(fv)
            fvts.append(fvt)

    obj = Obj()
    obj.vs = np.array(vs, np.float32)
    obj.vts = np.array(vts, np.float32)
    try:
        obj.fvs = np.array(fvs, int)
        obj.fvts = np.array(fvts, int)
    except ValueError:
        obj.fvs = fvs
        obj.fvts = fvts

    return obj


def write_obj(template_path, v: "Nx3", path, only_v=False):
    assert v.shape[1] == 3

    last_template = getattr(write_obj, "last_template", (None, None))

    if last_template[0] != template_path:
        obj = open(template_path).read().strip().split("\n")
        last_template = (template_path, obj)
        setattr(write_obj, "last_template", last_template)
    obj = last_template[1]
    vs = []
    for i in v:
        vs.append(f"v {i[0]:.6f} {i[1]:.6f} {i[2]:.6f}")
    if only_v:
        obj = "\n".join(vs)
    else:
        nvs = list(filter(lambda x: not x.startswith("v "), obj))
        obj = "\n".join(vs + nvs)

    open(path, "w").write(obj)


def read_camera_parameter(xml_path, para_folder, target_sensor_id):
    intrinsic_matrix = []
    extrinsic_matrix = []
    cameras = xml.dom.minidom.parse(xml_path)
    root = cameras.documentElement
    camera_list = root.getElementsByTagName('camera')
    sensor_index_list = []
    for camera in camera_list:
        # label = camera.getAttribute('label')
        sensor_id = camera.getAttribute('sensor_id')
        if sensor_id == target_sensor_id:
            sensor_id = camera.getAttribute('sensor_id')
            extrinsic_inv = np.array([float(x) for x in camera.getElementsByTagName('transform')[0].firstChild.data.split(' ')]).reshape(4, 4)
            extrinsic = np.linalg.inv(extrinsic_inv)
            extrinsic_matrix.append(extrinsic)
    # sensor_list = [element[0] for element in sensor_index_list]
    camera_sensor_list = root.getElementsByTagName('sensor')
    for sensor in camera_sensor_list:
        sensor_id = sensor.getAttribute('id')
        if sensor_id == target_sensor_id:
            f = float(sensor.getElementsByTagName('f')[0].firstChild.data)
            cx = float(sensor.getElementsByTagName('cx')[0].firstChild.data)
            cy = float(sensor.getElementsByTagName('cy')[0].firstChild.data)
            resolution = sensor.getElementsByTagName('resolution')[0]
            width = int(resolution.getAttribute('width'))
            height = int(resolution.getAttribute('height'))
            intrinsic = np.array([[f, 0, (width - 1) / 2 + cx, 0], [0, f, (height - 1) / 2 + cy, 0], [0, 0, 1, 0]])
            intrinsic_matrix.append(intrinsic)

    print("extrinsic number =", len(extrinsic_matrix))
    os.makedirs(para_folder, exist_ok=True)

    np.savetxt(os.path.join(para_folder, "intrinsic.txt"), np.vstack(intrinsic_matrix * 3))
    np.savetxt(os.path.join(para_folder, "extrinsic.txt"), np.vstack(extrinsic_matrix))


def translate_cal(para_folder, obj_path):
    ret = subprocess.run([
        r"translate_main.exe",
        "--extrinsic_txt_path", os.path.join(para_folder, "extrinsic.txt"),
        "--enlarge", "100",
        obj_path,
        os.path.join(para_folder, "translate.txt"),
    ])
    print(ret)
    if ret.returncode != 0:
        raise Exception("FAILED")


def translate_apply(ini, para_folder):
    obj_path, sub_inter_folder = ini
    # print(para_folder, obj_path, sub_inter_folder)
    os.makedirs(os.path.join(sub_inter_folder, "model"), exist_ok=True)
    ret = subprocess.run([
        r"translate_main.exe",
        "--translated_model_file_path", os.path.join(sub_inter_folder, "model", "translated.obj"),
        # "--enlarge", "100",
        obj_path,
        os.path.join(para_folder, "translate.txt"),
    ])
    print(ret)
    if ret.returncode != 0:
        raise Exception("FAILED")
    shutil.copy(os.path.join(sub_inter_folder, "model", "translated.obj"), os.path.join(sub_inter_folder, "model", "translated_origin.obj"))


def fit():
    import cv2
    import numpy as np

    small_shape = (1024, 1024)

    # def mypure(x, shape=None):
    #     if shape is not None:
    #         x = cv2.resize(x, shape)
    #     x = x.astype(np.float32)
    #     x[x == 0] = np.nan
    #     return (x - 32768) / 32767

    ref = mypure(cv2.imread(r"UV_to_normal_u16.png", -1), small_shape)
    my = mypure(cv2.imread(r"UV_diffuse_normal_merged_u16.png", -1), small_shape)

    cv2.imshow("ref", ref)
    cv2.imshow("my", my)

    samples = (np.isfinite(ref[..., 0]) & np.isfinite(my[..., 0])) &\
        (np.isfinite(ref[..., 1]) & np.isfinite(my[..., 1])) &\
        (np.isfinite(ref[..., 2]) & np.isfinite(my[..., 2]))
    cv2.imshow("samples", samples.astype(np.uint8) * 255)

    src = my[samples]
    tgt = ref[samples]

    print(np.where(np.isnan(src)))
    print(np.where(np.isnan(tgt)))

    A = np.concatenate((src, np.ones((src.shape[0], 1))), axis=1)
    b = tgt

    x, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

    print(x.shape)
    print(x)
    print(r / src.shape[0])

    my_ori = mypure(cv2.imread("UV_diffuse_normal_merged_u16.png", -1))
    src = my_ori.reshape(-1, 3)
    A = np.concatenate((src, np.ones((src.shape[0], 1))), axis=1)
    y = A @ x
    y /= np.linalg.norm(y, axis=1)[..., None]
    y = y.reshape(my_ori.shape[0], my_ori.shape[1], 3)
    cv2.imshow("my2", y)
    cv2.imwrite("my2.png", (y * 32767 + 32768).astype(np.uint16))

    cv2.waitKey()


@dive_path
def cal_normal(folder, inter_folder, para_folder):

    extrinsic_path = os.path.join(para_folder, "extrinsic.txt")
    extrinsics = np.loadtxt(extrinsic_path).reshape(3, 4, 4)

    photometric_corresponding_camera = 1
    photometric_corresponding_camera_to_normal_axis = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]])
    model_to_normal_axis = photometric_corresponding_camera_to_normal_axis @ extrinsics[
        photometric_corresponding_camera][:3, :3]

    point_to_self = np.array([0, 0, -1])

    views = []
    for extrinsic in extrinsics:
        view = model_to_normal_axis @ np.linalg.inv(extrinsic[:3, :3]) @ point_to_self
        view /= np.linalg.norm(view)
        # print(view)
        views.append(view)

    subfolders = [
        os.path.join(folder, "L"),
        os.path.join(folder, "M"),
        os.path.join(folder, "R"),
    ]

    for view, subfolder in zip(views, subfolders):
        savefolder = os.path.join(inter_folder, os.path.basename(folder), os.path.basename(subfolder))
        if os.path.exists(os.path.join(savefolder, "diffuse_normal.png")) and os.path.exists(os.path.join(savefolder, "specular_normal.png")) and os.path.exists(os.path.join(savefolder, "specular.png")) and os.path.exists(os.path.join(savefolder, "diffuse.png")):
            continue

        try:
            raw_images = [cv2.imread(i, -1)for i in sorted(glob.glob(os.path.join(subfolder, "*.tif")))[:14]]
            if len(raw_images) == 0:
                continue
            raw_c = raw_images[6]
            images = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).astype(np.float32) / 65535 for i in raw_images]
            x, xs, y, ys, z, zs, c, cs, ix, ixs, iy, iys, iz, izs = images
            diffuse_normal = np.concatenate((
                (x - ix)[..., None],
                (y - iy)[..., None],
                (z - iz)[..., None],
            ), axis=2)
            diffuse_normal /= np.linalg.norm(diffuse_normal, axis=2)[..., None]

            def pure(c):
                c[c < 0] = np.nan
                # c[c < 0] = -c[c < 0]
                return c

            pcs = pure(cs - c)

            pxs = pure(xs - x)
            pys = pure(ys - y)
            pzs = pure(zs - z)

            ipxs = pure(ixs - ix)
            ipys = pure(iys - iy)
            ipzs = pure(izs - iz)

            specular_normal = np.concatenate((
                (pxs - ipxs)[..., None],
                (pys - ipys)[..., None],
                (pzs - ipzs)[..., None],
            ), axis=2)
            specular_normal /= np.linalg.norm(specular_normal, axis=2)[..., None]
            specular_normal += view
            specular_normal /= np.linalg.norm(specular_normal, axis=2)[..., None]

            savefolder = os.path.join(inter_folder, os.path.basename(folder), os.path.basename(subfolder))
            os.makedirs(savefolder, exist_ok=True)

            cv2.imwrite(os.path.join(savefolder, "diffuse_normal.png"), (diffuse_normal[..., ::-1] * 32767 + 32768).astype(np.uint16))
            cv2.imwrite(os.path.join(savefolder, "specular_normal.png"), (specular_normal[..., ::-1] * 32767 + 32768).astype(np.uint16))
            cv2.imwrite(os.path.join(savefolder, "specular.png"), (pcs * 65535).astype(np.uint16))
            cv2.imwrite(os.path.join(savefolder, "diffuse.png"), (raw_c).astype(np.uint16))
        except Exception as e:
            print("fail on", subfolder)
            raise e


@dive_path
def cal_normal_dragon(folder, inter_folder, para_folder, subfolder_names=None):

    extrinsic_path = os.path.join(para_folder, "extrinsic.txt")
    extrinsics = np.loadtxt(extrinsic_path).reshape(3, 4, 4)

    photometric_corresponding_camera = 1
    photometric_corresponding_camera_to_normal_axis = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]])
    model_to_normal_axis = photometric_corresponding_camera_to_normal_axis @ extrinsics[
        photometric_corresponding_camera][:3, :3]

    point_to_self = np.array([0, 0, -1])

    views = []
    for extrinsic in extrinsics:
        view = model_to_normal_axis @ np.linalg.inv(extrinsic[:3, :3]) @ point_to_self
        view /= np.linalg.norm(view)
        # print(view)
        views.append(view)

    subfolders = [
        os.path.join(folder, "L"),
        os.path.join(folder, "M"),
        os.path.join(folder, "R"),
    ]
    if subfolder_names is not None:
        subfolders = [
            os.path.join(folder, name) for name in subfolder_names
        ]

    for k, (view, subfolder) in enumerate(zip(views, subfolders)):
        savefolder = os.path.join(inter_folder, os.path.basename(folder), os.path.basename(subfolder))
        if os.path.exists(os.path.join(savefolder, "diffuse_normal.png")) and os.path.exists(os.path.join(savefolder, "specular_normal.png")) and os.path.exists(os.path.join(savefolder, "specular.png")) and os.path.exists(os.path.join(savefolder, "diffuse.png")):
            continue

        try:
            # raw_images = [cv2.imread(i, -1)for i in sorted(glob.glob(os.path.join(subfolder, "*.tif")))[:38]]
            raw_images_path = sorted(glob.glob(os.path.join(subfolder, "*.tif")))[:38]
            if len(raw_images_path) == 0:
                continue
            # raw_c = raw_images[0]
            # images = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY).astype(np.float32) / 65535 for i in raw_images]

            (
                c1,
                xsl, xdl, ysl, ydl, zsl, zdl, xsln, xdln, ysln, ydln, zsln, zdln,
                xsm, xdm, ysm, ydm, zsm, zdm,
                c2,
                xsmn, xdmn, ysmn, ydmn, zsmn, zdmn,
                xsr, xdr, ysr, ydr, zsr, zdr, xsrn, xdrn, ysrn, ydrn, zsrn, zdrn,
            ) = raw_images_path

            xs = cv2.imread([xsl, xsm, xsr][k], -1).astype(np.float32) / 65535
            ys = cv2.imread([ysl, ysm, ysr][k], -1).astype(np.float32) / 65535
            zs = cv2.imread([zsl, zsm, zsr][k], -1).astype(np.float32) / 65535
            xsn = cv2.imread([xsln, xsmn, xsrn][k], -1).astype(np.float32) / 65535
            ysn = cv2.imread([ysln, ysmn, ysrn][k], -1).astype(np.float32) / 65535
            zsn = cv2.imread([zsln, zsmn, zsrn][k], -1).astype(np.float32) / 65535

            xd = cv2.imread([xdl, xdm, xdr][k], -1).astype(np.float32) / 65535
            yd = cv2.imread([ydl, ydm, ydr][k], -1).astype(np.float32) / 65535
            zd = cv2.imread([zdl, zdm, zdr][k], -1).astype(np.float32) / 65535
            xdn = cv2.imread([xdln, xdmn, xdrn][k], -1).astype(np.float32) / 65535
            ydn = cv2.imread([ydln, ydmn, ydrn][k], -1).astype(np.float32) / 65535
            zdn = cv2.imread([zdln, zdmn, zdrn][k], -1).astype(np.float32) / 65535

            cd = (xd + yd + zd + xdn + ydn + zdn) / 3

            xs = cv2.cvtColor(xs, cv2.COLOR_BGR2GRAY)
            ys = cv2.cvtColor(ys, cv2.COLOR_BGR2GRAY)
            zs = cv2.cvtColor(zs, cv2.COLOR_BGR2GRAY)
            xsn = cv2.cvtColor(xsn, cv2.COLOR_BGR2GRAY)
            ysn = cv2.cvtColor(ysn, cv2.COLOR_BGR2GRAY)
            zsn = cv2.cvtColor(zsn, cv2.COLOR_BGR2GRAY)
            xd = cv2.cvtColor(xd, cv2.COLOR_BGR2GRAY)
            yd = cv2.cvtColor(yd, cv2.COLOR_BGR2GRAY)
            zd = cv2.cvtColor(zd, cv2.COLOR_BGR2GRAY)
            xdn = cv2.cvtColor(xdn, cv2.COLOR_BGR2GRAY)
            ydn = cv2.cvtColor(ydn, cv2.COLOR_BGR2GRAY)
            zdn = cv2.cvtColor(zdn, cv2.COLOR_BGR2GRAY)

            diffuse_normal = np.concatenate((
                (xd - xdn)[..., None],
                (yd - ydn)[..., None],
                (zd - zdn)[..., None],
            ), axis=2)
            diffuse_normal /= np.linalg.norm(diffuse_normal, axis=2)[..., None]

            def pure(c):
                c[c < 0] = np.nan
                return c

            # pcs = pure(cs - c)

            pxs = pure(xs - xd)
            pys = pure(ys - yd)
            pzs = pure(zs - zd)

            ipxs = pure(xsn - xdn)
            ipys = pure(ysn - ydn)
            ipzs = pure(zsn - zdn)

            pcs = (pxs + pys + pzs + ipxs + ipys + ipzs) / 3

            specular_normal = np.concatenate((
                (pxs - ipxs)[..., None],
                (pys - ipys)[..., None],
                (pzs - ipzs)[..., None],
            ), axis=2)
            specular_normal /= np.linalg.norm(specular_normal, axis=2)[..., None]
            specular_normal += view
            specular_normal /= np.linalg.norm(specular_normal, axis=2)[..., None]

            savefolder = os.path.join(inter_folder, os.path.basename(folder), os.path.basename(subfolder))
            os.makedirs(savefolder, exist_ok=True)

            cv2.imwrite(os.path.join(savefolder, "diffuse_normal.png"), (diffuse_normal[..., ::-1] * 32767 + 32768).astype(np.uint16))
            cv2.imwrite(os.path.join(savefolder, "specular_normal.png"), (specular_normal[..., ::-1] * 32767 + 32768).astype(np.uint16))
            cv2.imwrite(os.path.join(savefolder, "specular.png"), (pcs * 65535).astype(np.uint16))
            cv2.imwrite(os.path.join(savefolder, "diffuse.png"), (cd.clip(0, 1) * 65535).astype(np.uint16))
        except Exception as e:
            print("fail on", subfolder)
            raise e


@dive_path
def custom_merge(folder, force=False):
    if not os.path.exists(os.path.join(folder, "UV_diffuse_normal_merged_u16.png")):
        return

    target_path = os.path.join(folder, f"merge.png")
    if os.path.exists(target_path) and not force:
        return

    ds = []
    ws = []

    scale = 1

    for i in [0, 1, 2]:
        a = read_normal(os.path.join(folder, f"UV_specular_normal_{i}_u16.png"), scale)

        a_blur = seamless_blur(a,)
        a_delta = a - a_blur

        cv2.imwrite(os.path.join(folder, f"a_delta.png"), a_delta * 127 + 128)
        print("fuck")
        return

        w = cv2.imread(os.path.join(folder, f"UV_weight_{i}.png"), cv2.IMREAD_GRAYSCALE) / 255 + 0.01

        ds.append(a_delta)
        ws.append(w)

    base = read_normal(os.path.join(folder, "UV_diffuse_normal_merged_u16.png"), scale)

    ds = np.array(ds)
    ws = np.array(ws)
    ws[np.isnan(ds[..., 0])] = 0
    ds[np.isnan(ds)] = 0

    r = np.sum(ds * ws[..., None], axis=0) + base
    total = normalize(r)

    # cv2.imshow(f"merge", total / 2 + 0.5)
    cv2.imwrite(target_path, (total * 32767 + 32768).astype(np.uint16))


@dive_path
def translate_any(folder):
    if os.path.exists(os.path.join(folder, "translated.obj")):
        return
    f = sorted(glob.glob(os.path.join(folder, "*.obj")))[0]
    obj = open(f).read()
    vs = [[float(i) for i in line.strip().split(" ")[1:]] + [1]
          for line in filter(lambda s:not s.startswith(("#", "mtllib", "usemtl")) and s != "", obj[:obj.find("vt")].strip().split("\n"))]
    vs = np.array(vs)
    translate = np.loadtxt(os.path.join(folder, "translate.txt"))[:3]
    nvs = (translate @ vs.T).T
    nvs = "\n".join(["v " + " ".join([f"{j:.6f}" for j in i]) for i in nvs])
    newobj = nvs + "\n" + obj[obj.find("vt"):]
    open(os.path.join(folder, "translated_origin.obj"), "w").write(newobj)
    open(os.path.join(folder, "translated.obj"), "w").write(newobj)


@dive_path
def translate_path(inpath, outfolder, translate_file_path):
    obj = open(inpath).read()
    vs = [[float(i) for i in line.strip().split(" ")[1:]] + [1]
          for line in filter(lambda s:not s.startswith(("#", "mtllib", "usemtl")) and s != "", obj[:obj.find("vt")].strip().split("\n"))]
    vs = np.array(vs)
    translate = np.loadtxt(translate_file_path)[:3]
    nvs = (translate @ vs.T).T
    nvs = "\n".join(["v " + " ".join([f"{j:.6f}" for j in i]) for i in nvs])
    newobj = nvs + "\n" + obj[obj.find("vt"):]
    open(os.path.join(outfolder, os.path.basename(inpath)), "w").write(newobj)


@dive_path
def modify_path(inpath, outfolder=None, source_file_path=None):
    obj = open(inpath).read()

    new = open(source_file_path).read()
    new = new[new.find("vt"):]

    old = open(inpath).read()
    old = old[:old.find("vt")]

    if outfolder is None:
        outfolder = os.path.dirname(inpath)
    print(inpath, source_file_path, outfolder)
    open(os.path.join(outfolder, os.path.basename(inpath)), "w").write(old + new)


@dive_path
def texgen(src_dst, size, para_folder, diffuse, specular, height, width, no_inter=True, no_bin=True, exe=r"texgen_main.exe", ksize=31 * 4, reserve=False):
    src, dst = src_dst
    # print(src)
    # if os.path.exists(os.path.join(folder, "process/UV_diffuse_merged.png")):
    #     return
    if os.path.exists(os.path.join(dst, "total_matrix.png")):
        print("skip", dst)
        return

    os.makedirs(dst, exist_ok=True)
    command = [
        exe,
        os.path.abspath(para_folder), os.path.abspath(src), os.path.abspath(dst), "--size", f"{size}",
        "--diffuse", f"{diffuse}",
        "--specular", f"{specular}",
        "--height", f"{height}",
        "--width", f"{width}",
        "--ksize", f"{ksize}",
    ]
    if no_inter:
        command.append("--no_inter")
    if no_bin:
        command.append("--no_bin")
    if reserve:
        command.append("--reserve")
    print(command)
    ret = subprocess.run(command)
    if ret.returncode != 0:
        print(ret)
        raise Exception("FAILED")


@dive_path
def combine(folder):
    ret = subprocess.run([
        r"combine_main.exe", folder,
        "--size", "4096",
        "--lambda", "0.5",
        "--eps", "0.001"
    ])
    print(ret, ret.returncode)
    if ret.returncode != 0:
        raise Exception(f"ret.returncode = {ret.returncode}")


def collect(inpath, outfolder):
    print(inpath)
    os.makedirs(outfolder, exist_ok=True)
    l = glob.glob(inpath)
    inpath = inpath.replace("\\", r"\\").replace("*", "(.*)")
    for path in tqdm.tqdm(l):

        result = re.match(inpath, path).groups()
        # print(inpath, path)
        # print(result)
        sub = "_".join(result)

        shutil.copy(path, os.path.join(outfolder, sub + path[path.rfind("."):]))


@dive_path
def dilate1(inpath):
    if os.path.exists(os.path.join(inpath, "UV_diffuse_merged_dilate1.png")):
        return

    assert os.path.exists(os.path.join(inpath, "UV_diffuse_merged.png")), os.path.join(inpath, "UV_diffuse_merged.png")

    diffuse = cv2.imread(os.path.join(inpath, "UV_diffuse_merged.png"), -1)[::-1]
    uvsize = diffuse.shape[0]

    area = np.fromfile(os.path.join(inpath, "UV_to_face.bin"), np.int).reshape(uvsize, uvsize)

    mask = (area >= 0).astype(np.uint16)

    kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])

    diffuse_sum = cv2.filter2D(diffuse, cv2.CV_16U, kernel)

    mask_sum = cv2.filter2D(mask, cv2.CV_16U, kernel)

    diffuse_new = diffuse_sum / mask_sum[..., None]
    diffuse_new[np.where(mask)] = diffuse[np.where(mask)]

    cv2.imwrite(os.path.join(inpath, "UV_diffuse_merged_dilate1.png"), diffuse_new.astype(np.uint16)[::-1])


@dive_path
def purge_obj(path):
    if os.path.basename(os.path.dirname(path)) == "deprecated":
        return
    print(path)

    backup = os.path.join(os.path.dirname(path), "deprecated")
    os.makedirs(backup, exist_ok=True)
    shutil.copy(path, backup)

    obj = open(path, encoding="utf8").read()
    obj = re.sub(r"(\d+/\d+)/\d+", lambda x: x.groups()[0], obj)
    obj = obj.strip().split("\n")
    obj = filter(lambda x: x.startswith("v ") or x.startswith("vt ") or x.startswith("f ") or x.startswith("mtllib "), obj)

    write = "\n".join(list(obj))

    open(path, "w").write(write)


def blendshape_transfer(blendshapes_folder, neutral_path, oscale, adaptive=False):
    blendshapes_path = sorted(glob.glob(os.path.join(blendshapes_folder, "*.obj")))

    def read_obj_vs(path, rest=None):
        obj = open(path).read()
        vs = []
        for v in re.findall(r"v .+", obj):
            vs.append([float(i) for i in v[1:].strip().split()])
        vs = np.array(vs)
        if rest:
            if obj.find("vt ") >= 0:
                rest = obj[obj.find("vt "):]
            else:
                rest = obj[obj.find("f "):]

        return vs, rest

    def save_obj_vs(path, vs, rest):
        open(path, "w").write(
            "\n".join([f"v {i[0]} {i[1]} {i[2]}" for i in vs]) + "\n" + rest
        )

    new_n, new_rest = read_obj_vs(neutral_path, True)

    tgt_path = os.path.join(os.path.dirname(neutral_path), f"{os.path.basename(neutral_path).split('.')[0]}_generated_bs")
    os.makedirs(tgt_path, exist_ok=True)

    if adaptive:
        scale = None
    else:
        scale = 1

    bs_n = None
    for path in tqdm.tqdm(blendshapes_path, "enumerating bs"):
        vs, _ = read_obj_vs(path)
        if bs_n is None:
            bs_n = vs
        # print(new_n.max(axis=0) - new_n.min(axis=0))
        # print(vs.max(axis=0) - vs.min(axis=0))
        if scale is None:
            scale = np.mean((new_n.max(axis=0) - new_n.min(axis=0)) / (bs_n.max(axis=0) - bs_n.min(axis=0)))
        new_vs = (vs - bs_n) * scale * oscale + new_n

        save_obj_vs(os.path.join(tgt_path, os.path.basename(path)), new_vs, new_rest)


def vt_vo_transfer(provide_vt_vo__path, provide_v__glob, output_dir, strict=False):

    provide_vt_vo = open(provide_vt_vo__path, encoding="utf8").read().strip().split("\n")
    provide_vt_vo__v = ([[float(j) for j in i[1:].strip().split(" ")] for i in filter(lambda x: x.startswith("v "), provide_vt_vo)])
    provide_vt_vo__vt = ([[float(j) for j in i.strip().split(" ")[1:]] for i in filter(lambda x: x.startswith("vt "), provide_vt_vo)])
    provide_vt_vo__dy = [{j for j in i[1:].strip().split(" ")} for i in filter(lambda x: x.startswith("f "), provide_vt_vo)]

    provide_vt_vo__vt2v = {}
    for i in provide_vt_vo__dy:
        for j in i:
            v, vt = [int(i) for i in j.split("/")]
            provide_vt_vo__vt2v[vt - 1] = v - 1

    print(sorted([tuple(i) for i in provide_vt_vo__vt])[:10])

    try:
        cache = pickle.load(open("vt_vo_transfer_cache.json", "rb"))
    except Exception as e:
        print(e)
        cache = {}

    provide_vt_vo__v2ignore = {}
    # provide_vt_vo__v2ignore = set(np.loadtxt("vert_id.txt", int))

    output = output_dir
    os.makedirs(output, exist_ok=True)

    key_assigned = {}

    for provide_v__path in sorted(glob.glob(provide_v__glob)):
        print(provide_v__path)
        # cache = {}
        provide_v = open(provide_v__path).read().strip().split("\n")
        provide_v__v = ([[float(j) for j in i[1:].strip().split(" ")] for i in filter(lambda x: x.startswith("v "), provide_v)])
        provide_v__vt = ([[float(j) for j in i[2:].strip().split(" ")[:2]] for i in filter(lambda x: x.startswith("vt "), provide_v)])
        provide_v__dy = [{j for j in i[1:].strip().split(" ")} for i in filter(lambda x: x.startswith("f "), provide_v)]

        provide_v__vt2v = {}
        for i in provide_v__dy:
            for j in i:
                v, vt = [int(i) for i in j.split("/")[:2]]
                provide_v__vt2v[vt - 1] = v - 1

        provide_v__vt_hash = {tuple(i): k for k, i in enumerate(provide_v__vt)}

        print(sorted(provide_v__vt_hash)[:10])

        provide_v__vt_np = []
        provide_v__vt_idx2np = []
        for i, key in enumerate(sorted(provide_v__vt_hash)):
            provide_v__vt_np.append(key)
            provide_v__vt_idx2np.append(key)
            assert key in provide_v__vt_hash
        provide_v__vt_np = np.array(provide_v__vt_np)
        print(provide_v__vt_np.shape)

        res = {}

        dist_list = [1e-3, 1e-2]
        if strict:
            dist_list = [1e-4]

        for dist in dist_list:
            for k, i in enumerate(tqdm.tqdm(provide_vt_vo__vt)):
                provide_vt_vo__vt_turn = tuple(i)
                if provide_vt_vo__vt_turn not in provide_v__vt_hash:
                    if provide_vt_vo__vt_turn not in cache:
                        # if provide_vt_vo__vt2v[k] in provide_vt_vo__v2ignore:
                        #     cache[provide_vt_vo__vt_turn] = None
                        # else:
                        # if provide_vt_vo__vt2v[k]==8:print("here is 8")
                        if True:
                            dists = np.linalg.norm(provide_v__vt_np - i, axis=1)
                            idx = np.argmin(dists)
                            provide_v__vt_turn = provide_v__vt_idx2np[idx]
                            if dists[idx] < dist and provide_v__vt_turn not in key_assigned:
                                cache[provide_vt_vo__vt_turn] = provide_v__vt_turn
                                key_assigned[provide_v__vt_turn] = provide_vt_vo__vt_turn
                            else:
                                print("warning: ignore", dists[idx])
                                if dist == dist_list[-1]:
                                    cache[provide_vt_vo__vt_turn] = None
                                # cache[turn] = key
                else:
                    pass
                    # if provide_vt_vo__vt2v[k]==8:print("here is 8 in")

                # assert turn in provide_v__vt_hash, turn

        for k, i in enumerate(tqdm.tqdm(provide_vt_vo__vt)):
            provide_vt_vo__vt_turn = tuple(i)
            if provide_vt_vo__vt_turn not in provide_v__vt_hash:
                provide_vt_vo__vt_turn = cache.get(provide_vt_vo__vt_turn, None)
            if provide_vt_vo__vt_turn is not None:
                res[provide_vt_vo__vt2v[k]] = provide_v__v[provide_v__vt2v[provide_v__vt_hash[provide_vt_vo__vt_turn]]]
            else:
                # print(f"{provide_vt_vo__vt2v[k]} not found")
                res[provide_vt_vo__vt2v[k]] = provide_vt_vo__v[provide_vt_vo__vt2v[k]]

        pickle.dump(cache, open("vt_vo_transfer_cache.json", "wb"))

        unique = {}

        j = 0
        k = 0
        for i in range(len(provide_vt_vo)):
            if provide_vt_vo[i].startswith("v "):
                provide_vt_vo[i] = f"v {res[j][0]} {res[j][1]} {res[j][2]}"
                j += 1

                # assert provide_vt_vo[i] not in unique, (i, provide_vt_vo[i], unique[provide_vt_vo[i]], provide_vt_vo[unique[provide_vt_vo[i]]])
                if provide_vt_vo[i] in unique:
                    print("WARNING! SAME LOCATION:", i, provide_vt_vo[i], unique[provide_vt_vo[i]], provide_vt_vo[unique[provide_vt_vo[i]]])
                unique[provide_vt_vo[i]] = i

        open(os.path.join(output, os.path.basename(provide_v__path)), "w").write("\n".join(list(filter(lambda x: len(x) > 0, provide_vt_vo))))


def topo_transfer(source_path, target_path, output_folder, map_path=None, source_glob=None):
    source_path = os.path.abspath(source_path)
    target_path = os.path.abspath(target_path)
    target = []
    vth = None
    for ln, line in enumerate(open(target_path).read().strip().split("\n")):
        if line.startswith("v "):
            if vth is None:
                vth = ln
        else:
            target.append(line)
    target_before = "\n".join(target[:vth])
    target_after = "\n".join(target[vth:])

    import trimesh
    import contrib.zyh_topo_transfer.topology_transfer as topology_transfer

    source_handle = topology_transfer.ObjHandle(source_path, triangulate=True)
    target_handle = topology_transfer.ObjHandle(target_path, triangulate=True)
    source = trimesh.Trimesh(vertices=source_handle.vertices, faces=source_handle.faces, process=False)
    target = trimesh.Trimesh(vertices=target_handle.vertices, faces=target_handle.faces, process=False)
    closest_face_ids, bCoords, d2S_ratios, boundary_indices, tangent_cp2tv = topology_transfer.computeTransfer(source, target)

    if map_path:
        topology_transfer.writePointOffset(map_path, closest_face_ids, bCoords, d2S_ratios, boundary_indices, tangent_cp2tv,
                                           target_handle.faces, target_handle.texcoords, target_handle.face_texs)
        np.savetxt(".".join(map_path.split(".")[:-1] + ["sourcefaces"] + map_path.split(".")[-1:]), source_handle.faces, fmt="%d")
    faces, tex_coords, face_tcs = target_handle.faces, target_handle.texcoords, target_handle.face_texs

    sources = glob.glob(os.path.dirname(source_path) + '/*.obj')
    if source_glob is not None:
        sources = glob.glob(source_glob)
    os.makedirs(output_folder, exist_ok=True)
    for source_file in sources:
        print(f"Transfer {source_file}")
        source_handle = topology_transfer.ObjHandle(source_file, triangulate=True)
        source = trimesh.Trimesh(vertices=source_handle.vertices, faces=source_handle.faces, process=False)
        target_verts = topology_transfer.applyTransfer(source, closest_face_ids, bCoords, d2S_ratios, boundary_indices, tangent_cp2tv)
        target_handle = topology_transfer.ObjHandle(vertices=target_verts, texcoords=tex_coords, faces=faces, face_texs=face_tcs,
                                                    mtl=source_handle.mtl, obj_material=source_handle.obj_material, triangulate=True)
        os.makedirs(output_folder, exist_ok=True)
        target_file = os.path.join(output_folder, os.path.basename(source_file))
        target_handle.write(target_file)
        final = "\n".join([target_before,
                           "\n".join(list(filter(lambda x: x.startswith("v "), open(target_file).read().strip().split("\n")))),
                           target_after])
        open(target_file, "w").write(final)


@dive_path
def flatten(path, alpha, kernel=None, blur=False):
    command = [
        "flatten_main",
        path,
        path[:-4] + "_n.png",
        "--alpha", f"{alpha}"
    ]
    if kernel:
        command.append("--kernel")
        command.append(f"{kernel}")
    if blur:
        command.append("--blurred_path")
        command.append(path[:-4] + "_b.png")

    subprocess.call(command)


@dive_path
def bake_tangent(path2):
    import contrib.qqw_bake_tangent.bake_tangent as bake_tangent
    obj_path, normal_input_path = path2

    bake_tangent.main(
        obj_path,
        normal_input_path,
        os.path.join(os.path.dirname(normal_input_path), "UV_to_normal_u16.png"),
        os.path.join(os.path.dirname(normal_input_path), os.path.basename(normal_input_path)[:-4] + "_tangent.png"),
    )


@dive_path
def folder_create_video(folder_path, fps):
    """
    Create a video from a folder of images
    """
    postfix = "png"
    img_glob = os.path.join(folder_path, f"*.{postfix}")
    if len(glob.glob(img_glob)) == 0:
        postfix = "jpg"
        img_glob = os.path.join(folder_path, f"*.{postfix}")
    video_path = os.path.dirname(img_glob).replace(".", "_").replace("\\", "_").replace("/", "_") + ".mp4"

    if len(glob.glob(img_glob)) == 0:
        return

    if os.path.exists(video_path):
        vmtime = os.path.getmtime(video_path)

        imtime = 0
        for img_path in glob.glob(img_glob):
            imtime = max(imtime, os.path.getmtime(img_path))

        if vmtime > imtime:
            return

    folder_path = folder_path.replace("\\", "/")

    cmd = f"""ffmpeg -r {fps} -pattern_type glob -i "{folder_path}/*.{postfix}" -pix_fmt yuv420p -q 0 {video_path} -y -loglevel error"""
    cmd = f"""ffmpeg -r {fps} -pattern_type glob -i "{folder_path}/*.{postfix}" -pix_fmt yuv420p -q 0 {video_path} -y"""
    subprocess.call(f"""wsl -e {cmd}""", shell=True)
    print(f"""generate {video_path}""")


@dive_path
def remove_fresnel(process_folder, para_folder):
    a = [0.17003849, 0.13474396, 0.05782015, 0.24853301, 0.13027675, 0.25858352]

    # translate = np.loadtxt(os.path.join(para_folder, "translate.txt"))
    extrinsics = np.loadtxt(os.path.join(para_folder, "extrinsic.txt"))

    point_to_self = np.array([[0, 0, -1]]).T
    photometric_corresponding_camera = 1
    photometric_corresponding_camera_to_normal_axis = np.array([0, 1, 0, 1, 0, 0, 0, 0, -1]).reshape(3, 3)
    model_to_normal_axis = photometric_corresponding_camera_to_normal_axis @ extrinsics[4:4 + 3, :3]

    views = []
    for i in range(0, 12, 4):
        view = model_to_normal_axis @ np.linalg.inv(extrinsics[i:i + 3, :3]) @ point_to_self
        view = view / np.linalg.norm(view)
        views.append(view)

    normal = os.path.join(process_folder, "UV_to_normal_u16.png")
    normal = read_normal(normal, use_xyz=True)

    normal[:, :, 1] = 0
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)

    defresneled = []
    weight = np.array([cv2.imread(i, -1) / 65535 for i in sorted(glob.glob(os.path.join(process_folder, "UV_weight_*.png")))]) + 1e-4
    specular = [cv2.imread(i, -1) / 65535 for i in sorted(glob.glob(os.path.join(process_folder, "UV_specular_*_u16.png")))]

    for t in range(3):
        z = (normal * views[t].T).sum(axis=2)
        z[z < 0] = 0
        angle = np.arccos(z)
        angle01 = angle / (np.pi / 2)

        scale = 0
        xx = 1
        for i in range(len(a)):
            scale += a[i] * xx
            xx *= angle01

        defresneled.append(specular[t] / scale)

    merge = (defresneled * weight).sum(axis=0) / weight.sum(axis=0)
    cv2.imwrite(os.path.join(process_folder, "UV_specular_merged_defresnel.png"), (merge * 65535).clip(0, 65535).astype(np.uint16))


@dive_path
def lr_balance(path):
    img_origin = cv2.imread(path)
    img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
    img_origin = cv2.resize(img_origin, (1024, 1024), interpolation=cv2.INTER_AREA) / 255
    img = img_origin.copy()

    img[img == 0] = np.nan
    img[img < 0.05] = np.nan

    ksize = 32

    N = 4

    img_ds = []

    for i in range(N):
        img_b = seamless_blur(img, ksize * 2**i + 1)
        img_d = img - img_b
        img_ds.append(img_d)
        img = img_b

    num = (np.isfinite(img).astype(np.float32) + np.isfinite(img[:, ::-1]).astype(np.float32))
    img[np.isnan(img)] = 0

    img = (img + img[:, ::-1]) / num

    for i in range(N):
        img = img + img_ds[-i - 1]

    scale = img / img_origin
    scale[np.isnan(scale)] = 1
    scale[scale == 0] = 1

    scale = scale.clip(0.9, 1.1)

    cv2.imwrite("scale.png", scale * 128)

    img_raw = cv2.imread(path, -1)
    scale = cv2.resize(scale, (img_raw.shape[1], img_raw.shape[0]))

    if len(img_raw.shape) >= 3:
        scale = scale[:, :, None]

    img_scaled = img_raw.astype(np.float32) * scale

    if img_raw.dtype == np.uint8:
        img_scaled = img_scaled.clip(0, 255)
    elif img_raw.dtype == np.uint16:
        img_scaled = img_scaled.clip(0, 65535)

    img_scaled = img_scaled.astype(img_raw.dtype)

    cv2.imwrite(os.path.join(os.path.dirname(path), ".".join(os.path.basename(path).split(".")[:-1] + ["lrb"] + os.path.basename(path).split(".")[-1:])), img_scaled)


def align_vertices(src, dst, scale=False):
    """
    Vx3
    return: src_aligned, R, T, [scale]
    """
    assert src.shape == dst.shape, (src.shape, dst.shape)

    P = src.T
    Srecon = dst.T

    P = P.copy()
    Srecon = Srecon.copy()

    S_0 = Srecon.copy()
    S_0_mean = np.mean(S_0, axis=1)[..., None]
    S_0 -= S_0_mean

    S_0_std = ((S_0**2).sum() / S_0.shape[1])**0.5

    P_0 = P.copy()
    P_0_mean = np.mean(P_0, axis=1)[..., None]
    P_0 -= P_0_mean

    P_0_std = ((P_0**2).sum() / P_0.shape[1])**0.5

    W = S_0@P_0.T
    U, sigma, VT = np.linalg.svd(W)
    R0 = U@VT
    if scale:
        R = R0 / P_0_std * S_0_std
    else:
        R = R0
    T = S_0_mean - R@P_0_mean

    P_aligned = R@src.T + T

    if scale:
        return P_aligned.T, R, T, 1 / P_0_std * S_0_std

    return P_aligned.T, R, T


def iterative_solve_blendshapes(source, shapes_mean, shapes_diff, scale=False, min_max=(0, 2)):
    """
    source: [Nv, 3]
    shapes_mean: [1, Nv, 3]
    shapes_diff: [M, Nv, 3]

    Ax =

    return: x
    """

    M = shapes_diff.shape[0]
    x_last = np.zeros(M)

    last_dist = np.inf

    for i in range(20):
        recon = (x_last@shapes_diff.reshape(M, -1) + shapes_mean.reshape(-1)).reshape(-1, 3)
        if scale:
            source_aligned, R, T, scale_ret = align_vertices(source, recon, scale=scale)
        else:
            source_aligned, R, T = align_vertices(source, recon, scale=scale)

        dist = (np.linalg.norm(source_aligned - recon)**2 / len(recon))**0.5

        AA = shapes_diff.reshape(M, -1).T
        BB = source_aligned.reshape(-1) - shapes_mean.reshape(-1)

        XT = opt.lsq_linear(AA, BB, min_max)
        x_last = XT.x
        print("rmse =", dist)
        if last_dist - dist < 1e-4:
            break
        last_dist = dist

    if scale:
        return x_last, source_aligned, R, T, scale_ret
    return x_last, source_aligned, R, T


def zip_glob(*globs, enum=False):

    globs = [sorted(glob.glob(i)) for i in globs]

    lengths = [len(i) for i in globs]
    if max(lengths) != min(lengths):
        warnings.warn("globs have different lengths")

    globs[0] = tqdm.tqdm(globs[0])

    if enum:
        return enumerate(zip(*globs))

    return zip(*globs)


def render_mesh(texture_input, mesh, resolution, intrinsic3x4, extrinsic: "4x4", affine: "4x4" = np.identity(4)):

    import contrib.qqw_bake_tangent.render_mesh as render_mesh
    return render_mesh.render_mesh(texture_input, mesh, resolution, intrinsic3x4, extrinsic, affine)


def main():
    parser = argparse.ArgumentParser(description='zlw tool box')
    subparsers = parser.add_subparsers(dest="func_name", required=True)

    camera_parser = subparsers.add_parser("camera")
    camera_parser.add_argument("xml_path", type=str)
    camera_parser.add_argument("para_folder", type=str)
    camera_parser.add_argument("--target_sensor_id", type=str, default="1")

    translate_cal_parser = subparsers.add_parser("translate_cal")
    translate_cal_parser.add_argument("para_folder", type=str)
    translate_cal_parser.add_argument("obj_path", type=str)

    cal_parser = subparsers.add_parser("cal")
    cal_parser.add_argument("para_folder", type=str)
    cal_parser.add_argument("inter_folder", type=str)
    cal_parser.add_argument("--dragon", default=False, action="store_const", const=True)
    cal_parser.add_argument("--subfolder_names", default=None, type=str)
    cal_parser.add_argument("paths", type=str, nargs="+")

    translate_apply_parser = subparsers.add_parser("translate_apply")
    translate_apply_parser.add_argument("para_folder", type=str)
    translate_apply_parser.add_argument("sub_inter_folders", type=str)
    translate_apply_parser.add_argument("paths", type=str, nargs="+")

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("paths", type=str, nargs="+")

    modify_path_parser = subparsers.add_parser("modify_path")
    modify_path_parser.add_argument("reference", type=str)
    modify_path_parser.add_argument("paths", type=str, nargs="+")

    texgen_parser = subparsers.add_parser("texgen")
    texgen_parser.add_argument("para_folder", type=str)
    texgen_parser.add_argument("sub_inter_folders", type=str)
    texgen_parser.add_argument("process_folder", type=str)
    texgen_parser.add_argument("size", type=int)
    texgen_parser.add_argument("--diffuse", type=float, default=1)
    texgen_parser.add_argument("--specular", type=float, default=1)
    texgen_parser.add_argument("--height", type=int, default=2720)
    texgen_parser.add_argument("--width", type=int, default=4800)
    texgen_parser.add_argument("--ksize", type=float, default=31 * 4)
    texgen_parser.add_argument("--inter", default=False, action='store_true')
    texgen_parser.add_argument("--bin", default=False, action='store_true')
    texgen_parser.add_argument("--reserve", default=False, action='store_true')
    texgen_parser.add_argument("--exe", type=str, default=r"texgen_main.exe", help=r"for gpu use: E:\FACESCAN_20220204UPDATE\standard_texture_generation\sur_render_zlw\build\test\Release\texgen_main.exe")

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("paths", type=str)
    collect_parser.add_argument("outfolder", type=str)

    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("paths", type=str, nargs="+")

    dilate1_parser = subparsers.add_parser("dilate1")
    dilate1_parser.add_argument("paths", type=str, nargs="+")

    purge_obj_parser = subparsers.add_parser("purge_obj")
    purge_obj_parser.add_argument("folders", type=str, nargs="+")

    blendshape_transfer_parser = subparsers.add_parser("blendshape_transfer")
    blendshape_transfer_parser.add_argument("blendshapes_folder", type=str)
    blendshape_transfer_parser.add_argument("neutral_path", type=str)
    blendshape_transfer_parser.add_argument("--scale", type=float, default=1)
    blendshape_transfer_parser.add_argument("--adaptive", default=False, action='store_true')

    vt_vo_transfer_parser = subparsers.add_parser("vt_vo_transfer")
    vt_vo_transfer_parser.add_argument("provide_vt_vo__path", type=str)
    vt_vo_transfer_parser.add_argument("provide_v__glob", type=str)
    vt_vo_transfer_parser.add_argument("output_dir", type=str)
    vt_vo_transfer_parser.add_argument("--strict", default=False, action='store_true')

    topo_transfer_parser = subparsers.add_parser("topo_transfer")
    topo_transfer_parser.add_argument("source_path", type=str)
    topo_transfer_parser.add_argument("target_path", type=str)
    topo_transfer_parser.add_argument("output_folder", type=str)
    topo_transfer_parser.add_argument("--map_path", type=str, default=None)
    topo_transfer_parser.add_argument("--source_glob", type=str, default=None)

    flatten_parser = subparsers.add_parser("flatten")
    flatten_parser.add_argument("paths", type=str, nargs="+")
    flatten_parser.add_argument("--alpha", type=float, default=1)
    flatten_parser.add_argument("--kernel", type=float, default=None)
    flatten_parser.add_argument("--blur", default=False, action='store_true')

    bake_tangent_parser = subparsers.add_parser("bake_tangent")
    bake_tangent_parser.add_argument("inter_folders", type=str)
    bake_tangent_parser.add_argument("process_folders", type=str)

    video_parser = subparsers.add_parser("video")
    video_parser.add_argument("paths", type=str, nargs="+")
    video_parser.add_argument("--fps", type=int, default=30)

    fresnel_parser = subparsers.add_parser("fresnel")
    fresnel_parser.add_argument("para_folder", type=str)
    fresnel_parser.add_argument("paths", type=str, nargs="+")

    lr_balance_parser = subparsers.add_parser("lr_balance")
    lr_balance_parser.add_argument("paths", type=str, nargs="+")

    parser.add_argument('-j', type=int, help='number of threads', default=0)
    args = parser.parse_args()

    print("current function is", args.func_name)

    targets = None
    prefer_j = 4

    if hasattr(args, "paths"):
        targets = []
        for f in args.paths:
            targets += glob.glob(f)
        targets = sorted(targets)
        print("targets =", targets)

    if args.func_name == "cal":
        func = functools.partial(cal_normal, inter_folder=args.inter_folder, para_folder=args.para_folder)
        if args.dragon:
            print("use dragon")
            func = functools.partial(cal_normal_dragon, inter_folder=args.inter_folder, para_folder=args.para_folder, subfolder_names=args.subfolder_names)

    elif args.func_name == "translate_apply":

        targets2 = glob.glob(args.sub_inter_folders)
        targets2 = sorted(targets2)
        print("targets2 =", targets2)
        assert len(targets) == len(targets2), (len(targets), len(targets2))
        targets = zip(targets, targets2)

        func = functools.partial(translate_apply, para_folder=args.para_folder)

    elif args.func_name == "texgen":
        prefer_j = 1
        targets = sorted(glob.glob(args.sub_inter_folders))
        targets = zip(targets, [os.path.join(args.process_folder, os.path.basename(i)) for i in targets])

        func = functools.partial(texgen, size=args.size, para_folder=args.para_folder, diffuse=args.diffuse, specular=args.specular, height=args.height, width=args.width, no_inter=not args.inter, no_bin=not args.bin, exe=args.exe, ksize=args.ksize, reserve=args.reserve)

    elif args.func_name == "merge":
        func = custom_merge

    elif args.func_name == "collect":
        collect(args.paths, args.outfolder)
        exit()

    elif args.func_name == "modify_path":
        func = functools.partial(modify_path, source_file_path=args.reference)

    elif args.func_name == "combine":
        func = combine
        args.j = 1

    elif args.func_name == "camera":
        read_camera_parameter(xml_path=args.xml_path, para_folder=args.para_folder, target_sensor_id=args.target_sensor_id)
        exit()

    elif args.func_name == "translate_cal":
        translate_cal(args.para_folder, args.obj_path)
        exit()

    elif args.func_name == "dilate1":
        func = dilate1

    elif args.func_name == "purge_obj":
        func = purge_obj
        targets = []
        for f in args.folders:
            if not f.endswith(".obj"):
                f = os.path.join(f, "*.obj")
            targets += glob.glob(f)
        targets = sorted(targets)
        print("targets =", targets)

    elif args.func_name == "blendshape_transfer":
        blendshape_transfer(args.blendshapes_folder, args.neutral_path, args.scale, adaptive=args.adaptive)
        exit()

    elif args.func_name == "vt_vo_transfer":
        vt_vo_transfer(args.provide_vt_vo__path, args.provide_v__glob, args.output_dir, args.strict)
        exit()

    elif args.func_name == "topo_transfer":
        topo_transfer(args.source_path, args.target_path, args.output_folder, args.map_path, args.source_glob)
        exit()

    elif args.func_name == "flatten":
        prefer_j = 1
        func = functools.partial(flatten, alpha=args.alpha, kernel=args.kernel, blur=args.blur)

    elif args.func_name == "bake_tangent":
        func = bake_tangent
        targets = []

        obj_paths = sorted(glob.glob(os.path.join(args.inter_folders, "model", "translated.obj")))
        normal_paths = sorted(glob.glob(os.path.join(args.process_folders, "total_matrix.png")))

        assert len(obj_paths) == len(normal_paths)

        targets = zip(obj_paths, normal_paths)

    elif args.func_name == "video":
        func = functools.partial(folder_create_video, fps=args.fps)

    elif args.func_name == "fresnel":
        func = functools.partial(remove_fresnel, para_folder=args.para_folder)

    elif args.func_name == "lr_balance":
        func = lr_balance

    if args.j == 0:
        args.j = prefer_j

    print("args.j =", args.j)
    with ThreadPool(args.j) as tp:
        tp.map(func, targets)
        print("done")


if __name__ == "__main__":
    main()
