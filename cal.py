import re
import numpy as np
import os
import glob
import cv2
import shutil
import subprocess
import tqdm
import multiprocessing
import zlw


#flame_vs_delta = zlw.read_obj("testing/00000.obj", True).vs - zlw.read_obj("testing/flame_neutral.obj", True).vs
#usc_neutral_vs = zlw.read_obj(r"Y:\ProjectHH_ProcessingKit\Wrap_Template\20211231_USC_bs_deemosilization\USC_BS_14062\000_generic_neutral_mesh.obj", True).vs

record = open("map.txt").read().strip().split("\n")


sourcesfaces = np.loadtxt("map.sourcefaces.txt", int)


face_ids = []
bcoords = []

for row in record[1:1 + 3931]:
    row = row.strip().split()
    face_ids.append(int(row[0]))
    bcoords.append([float(i) for i in row[1:4]])

face_ids = np.array(face_ids)
bcoords = np.array(bcoords)


np.save("face_ids.npy", face_ids)
np.save("bcoords.npy", bcoords)
