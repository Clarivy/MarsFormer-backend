import os
import numpy as np


class MeshData(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.vertex_format = [
            ('v_pos', 3, 'float'),
            ('v_normal', 3, 'float'),
            ('v_tc0', 2, 'float'),
            ('v_ambient', 3, 'float'),
            ('v_diffuse', 3, 'float'),
            ('v_specular', 3, 'float')
        ]
        self.vertices = []
        self.indices = []
        # Default basic material of mesh object
        self.diffuse_color = np.array([1.0, 1.0, 1.0])
        self.ambient_color = np.array([1.0, 1.0, 1.0])
        self.specular_color = np.array([1.0, 1.0, 1.0])
        self.specular_coefficent = 16.0
        self.transparency = 1.0

    def set_materials(self, mtl_dict):
        self.diffuse_color = mtl_dict.get('Kd', self.diffuse_color)
        self.diffuse_color = np.array([float(v) for v in self.diffuse_color])
        self.ambient_color = mtl_dict.get('Ka', self.ambient_color)
        self.ambient_color = np.array([float(v) for v in self.ambient_color])
        self.specular_color = mtl_dict.get('Ks', self.specular_color)
        self.specular_color = np.array([float(v) for v in self.specular_color])
        self.specular_coefficent = float(
            mtl_dict.get('Ns', self.specular_coefficent))
        transparency = mtl_dict.get('d')
        if not transparency:
            transparency = 1.0 - float(mtl_dict.get('Tr', 0))
        self.transparency = float(transparency)


class MTL(object):
    def __init__(self, filename):
        self.contents = {}
        self.filename = filename
        if not os.path.exists(filename):
            return
        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'newmtl':
                mtl = self.contents[values[1]] = {}
            elif mtl is None:
                raise ValueError("mtl file doesn't start with newmtl stmt")
            if len(values[1:]) > 1:
                mtl[values[0]] = values[1:]
            else:
                mtl[values[0]] = values[1]

    def __getitem__(self, key):
        return self.contents[key]

    def get(self, key, default=None):
        return self.contents.get(key, default)


class ObjHandle(object):
    """  """

    def __init__(self, filename=None, swapyz=False, delimiter=None,
                 vertices=None, normals=None, texcoords=None, faces=None, face_norms=None, face_texs=None,
                 obj_material=None, mtl=None, triangulate=False):
        if filename is not None:
            self.load(filename, swapyz, delimiter, triangulate=triangulate)
        else:
            self.vertices = np.array(vertices) if vertices is not None else np.array([])
            self.normals = np.array(normals) if normals is not None else np.array([])
            self.texcoords = np.array(texcoords) if texcoords is not None else np.array([])
            self.faces = np.array(faces) if faces is not None else np.array([[]])
            self.face_norms = np.array(face_norms) if face_norms is not None else np.empty((self.faces.shape[0], 0))
            self.face_texs = np.array(face_texs) if face_texs is not None else np.empty((self.faces.shape[0], 0))
            if swapyz:
                self.vertices = self.vertices[: [0, 2, 1]]
                self.normals = self.normals[: [0, 2, 1]]

            self.objects = {}
            self._current_object = None
            self.obj_material = obj_material
            self.mtl = mtl

    def load(self, filename, swapyz=False, delimiter=None, triangulate=False):
        """Loads a Wavefront OBJ file. """
        self.objects = {}
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.face_norms = []
        self.face_texs = []

        self._current_object = None
        self.mtl = None
        self.obj_material = None

        for line in open(filename, "r", encoding="utf8"):
            if delimiter is not None and delimiter == "# object" and "# object" in line:
                if self._current_object:
                    self.finish_object()
                self._current_object = line.split()[2]
            if line.startswith('#'):
                continue
            if line.startswith('s'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'o' and len(self.vertices) != 0:
                self.finish_object()
                self._current_object = values[1]
            elif values[0] == 'mtllib':
                # load materials file here
                self.mtl = MTL(values[1])
            elif values[0] in ('usemtl', 'usemat'):
                self.obj_material = values[1]
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = [v[0], v[2], v[1]]
                    #v = [v[1], v[2], v[0]]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = [v[0], v[2], v[1]]
                    #v = [v[1], v[2], v[0]]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] == 'f':
                if not triangulate or len(values) == 4:
                    face = []
                    texcoords = []
                    norms = []
                    for v in values[1:]:
                        w = v.split('/')
                        face.append(int(w[0]) - 1)
                        if len(w) >= 2 and len(w[1]) > 0:
                            if '//' not in v:
                                texcoords.append(int(w[1]) - 1)
                            else:
                                norms.append(int(w[1]) - 1)
                        if len(w) >= 3 and len(w[2]) > 0:
                            norms.append(int(w[2]) - 1)
                    self.faces.append(face)
                    self.face_norms.append(norms)
                    self.face_texs.append(texcoords)
                else:

                    face = []
                    texcoords = []
                    norms = []
                    for v in values[1:4]:
                        w = v.split('/')
                        face.append(int(w[0]) - 1)
                        if len(w) >= 2 and len(w[1]) > 0:
                            if '//' not in v:
                                texcoords.append(int(w[1]) - 1)
                            else:
                                norms.append(int(w[1]) - 1)
                        if len(w) >= 3 and len(w[2]) > 0:
                            norms.append(int(w[2]) - 1)
                    self.faces.append(face)
                    self.face_norms.append(norms)
                    self.face_texs.append(texcoords)

                    face = []
                    texcoords = []
                    norms = []
                    for v in [values[1], values[3], values[4]]:
                        w = v.split('/')
                        face.append(int(w[0]) - 1)
                        if len(w) >= 2 and len(w[1]) > 0:
                            if '//' not in v:
                                texcoords.append(int(w[1]) - 1)
                            else:
                                norms.append(int(w[1]) - 1)
                        if len(w) >= 3 and len(w[2]) > 0:
                            norms.append(int(w[2]) - 1)
                    self.faces.append(face)
                    self.face_norms.append(norms)
                    self.face_texs.append(texcoords)

        self.finish_object()

    def finish_object(self):
        self.vertices = np.array(self.vertices)
        self.normals = np.array(self.normals)
        self.texcoords = np.array(self.texcoords)
        self.faces = np.array(self.faces)
        self.face_norms = np.array(self.face_norms)
        self.face_texs = np.array(self.face_texs)
        if self._current_object is None:
            return

        mesh = MeshData()
        idx = 0
        material = self.mtl.get(self.obj_material)
        if material:
            mesh.set_materials(material)
        for verts, norms, tcs in zip(self.faces, self.face_norms, self.face_texs):
            # verts = f[0]
            # norms = f[1]
            # tcs = f[2]
            for i in range(3):
                # get normal components
                n = np.array([0.0, 0.0, 0.0])
                if norms is not None:
                    n = self.normals[norms[i] - 1]

                # get texture coordinate components
                t = np.array([0.0, 0.0])
                if tcs is not None:
                    t = self.texcoords[tcs[i] - 1]

                # get vertex components
                v = self.vertices[verts[i] - 1]

                data = [v[0], v[1], v[2], n[0], n[1], n[2], t[0], t[1]]
                mesh.vertices.extend(data)

                # add material info in the vertice
                mesh.vertices.extend(
                    mesh.ambient_color +
                    mesh.diffuse_color +
                    mesh.specular_color
                )

            tri = [idx, idx + 1, idx + 2]
            mesh.indices.extend(tri)
            idx += 3

        mesh.vertices = np.array(mesh.vertices)
        mesh.indices = np.array(mesh.indices)

        self.objects[self._current_object] = mesh
        # mesh.calculate_normals()
        self.faces = []

    def write(self, filename, swapyz=False):
        # NOTE: NO CONSIDERATION FOR DELIMITER NOW
        with open(filename, 'w') as f:
            if self.mtl is not None:
                f.write(f"mtllib {self.mtl.filename}\n")
            for v in self.vertices:
                if swapyz:
                    v = v[0], v[2], v[1]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            if self.texcoords.size != 0:
                for vt in self.texcoords:
                    f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
            if self.normals.size != 0:
                for vn in self.normals:
                    if swapyz:
                        vn = vn[0], vn[2], vn[1]
                    f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
            if self.obj_material is not None:
                f.write(f"usemtl {self.obj_material}\n")
            for verts, norms, tcs in zip(self.faces, self.face_norms, self.face_texs):
                f.write("f ")
                for i in range(verts.shape[0]):
                    f.write(f"{verts[i] + 1}")

                    if tcs.size != 0:
                        f.write(f"/{tcs[i] + 1}")
                    elif norms.size != 0:
                        f.write("/")
                    if norms.size != 0:
                        f.write(f"/{norms[i] + 1}")

                    if i != verts.shape[0] - 1:
                        f.write(" ")
                f.write("\n")
