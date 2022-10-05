import os
import numpy as np

def triangleFrame(v1, v2, v3):
    # Return triangle frame matrix of vertices V1, V2, V3
    # The third component could be expressed as V4 - V1
    if len(v1.shape) == 2:
        return np.stack([v2 - v1, v3 - v1, np.cross(v2 - v1, v3 - v1)], axis=1)
    else:
        return np.stack([v2 - v1, v3 - v1, np.cross(v2 - v1, v3 - v1)], axis=0)

def getFileList(root, ext, recursive=False):
    files = []
    dirs = os.listdir(root)
    while len(dirs) > 0:
        path = dirs.pop()
        fullname = os.path.join(root, path)
        if os.path.isfile(fullname) and fullname.endswith(ext):
            files.append(path)
        elif recursive and os.path.isdir(fullname):
            for s in os.listdir(fullname):
                newDir = os.path.join(path, s)
                dirs.append(newDir)
    files = sorted(files)
    return files