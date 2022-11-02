import numpy as np

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
