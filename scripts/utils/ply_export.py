#!/usr/bin/env python

import numpy as np

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, gray):
    verts = verts.reshape(-1, 3)
    gray = gray.reshape(-1, 1)
    verts = np.hstack([verts, gray, gray, gray])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
