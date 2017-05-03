#!/usr/bin/env python

import numpy as np
import cv2
import glob
import argparse
import os

import pykitti # version 0.1.2

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

if __name__ == '__main__':
    print('loading')

    dataset = pykitti.odometry(
        '/home/wolf/kitti/dataset',
        '00',
        frame_range=range(121, 150, 5)
    )

    dataset.load_calib()
    dataset.load_timestamps()
    dataset.load_poses()
    dataset.load_gray()

    window_size = 3
    min_disp = 16
    num_disp = 112 - min_disp

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=16,
        P1=8*3*window_size**2,
        P2=32*3*window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    for pair in dataset.gray:
        print('computing disparity')

        disp = stereo.compute(
            np.uint8(pair.left * 255.),
            np.uint8(pair.right * 255.)
        ).astype(np.float32) / 16.0

        calibration_matrix = dataset.calib.K_cam0 # left? (they're all the same here)
        # import pdb; pdb.set_trace()

        Q = np.float32([
            [1, 0, 0, -calibration_matrix[0, 2]],
            [0,-1, 0,  calibration_matrix[1, 2]], # turn points 180 deg around x-axis,
            [0, 0, 0, -calibration_matrix[0, 0]], # so that y-axis looks up
            [0, 0, 1, 0]
        ])
        points = cv2.reprojectImageTo3D(disp, Q)
        mask = disp > disp.min()
        out_points = points[mask]
        out_gray = np.uint8(pair.left * 255.)[mask]
        out_file = 'out.ply'

        write_ply(out_file, out_points, out_gray)

        break

