import os
import glob
import numpy as np

class OBJ:
    def __init__(self, v, vt, f):
        self.v = np.array(v)
        self.vt = np.array(vt)
        self.f = np.array(f)
    def get_v_num(self):
        return self.v.shape[0]
    def get_vt_num(self):
        return self.vt.shape[0]
    def get_f_num(self):
        return len(self.f)
    def get_v(self, index):
        return self.v[index]
    def get_f(self, index):
        return self.f[index]
    def get_v_neighbor(self):
        v_neighbor = [set() for i in range(self.get_v_num())]
        for i in range(self.get_f_num()):
            f = self.get_f(i)
            for j in range(len(f)):
                if j != len(f)-1:
                    v_neighbor[f[j][0]].add(f[j+1][0])
                    v_neighbor[f[j+1][0]].add(f[j][0])
                else:
                    v_neighbor[f[j][0]].add(f[0][0])
                    v_neighbor[f[0][0]].add(f[j][0])
        return v_neighbor
    def get_length(self, v0, v1):
        p0 = self.v[v0]
        p1 = self.v[v1]
        return np.linalg.norm(p0 - p1)
    def get_vt(self):
        return self.vt
    def get_uv_mapping(self):
        uv_mapping = [set() for i in range(self.get_v_num())]
        for i in range(self.get_f_num()):
            f = self.get_f(i)
            for j in range(len(f)):
                uv_mapping[f[j][0]].add(f[j][1])
        return uv_mapping


def read_obj(obj_file):
    v = []
    vt = []
    face = []
    with open(obj_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split()
            if len(data) == 0:
                continue
            if data[0] == 'v':
                v.append((float(data[1]), float(data[2]), float(data[3])))
            elif data[0] == 'vt':
                vt.append((float(data[1]), float(data[2])))
            elif data[0] == 'f':
                if len(data) == 5:
                    data1 = data[1].split('/')
                    data2 = data[2].split('/')
                    data3 = data[3].split('/')
                    data4 = data[4].split('/')
                    face.append(((int(data1[0])-1, int(data1[1])-1), (int(data2[0])-1, int(data2[1])-1), (int(data3[0])-1, int(data3[1])-1)))
                    face.append(((int(data3[0])-1, int(data3[1])-1), (int(data4[0])-1, int(data4[1])-1), (int(data1[0])-1, int(data1[1])-1)))
                elif len(data) == 4:
                    data1 = data[1].split('/')
                    data2 = data[2].split('/')
                    data3 = data[3].split('/')
                    face.append(((int(data1[0])-1, int(data1[1])-1), (int(data2[0])-1, int(data2[1])-1), (int(data3[0])-1, int(data3[1])-1)))
    return OBJ(v, vt, face)


def read_blendshape(path):
    files = glob.glob(os.path.join(path, "*.obj"))
    objs = [read_obj(i) for i in files]
    return objs, files

