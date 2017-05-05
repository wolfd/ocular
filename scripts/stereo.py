#!/usr/bin/env python

import numpy as np
import cv2
import glob
import argparse
import os

import pykitti # version 0.1.2

import matplotlib.pyplot as plt

from utils.ply_export import write_ply
from features.sift_image import SIFTImage

from utils.camera_transform_guesser import rigid_transform_3D

DEBUG_SIFT_3D = False
DEBUG_STEREO_3D = False
SET_FIRST_POSE = True

if __name__ == '__main__':
    print('loading')

    dataset = pykitti.odometry(
        '/home/wolf/kitti/dataset',
        '00',
        frame_range=range(85, 430, 3)
    )

    dataset.load_calib()
    dataset.load_timestamps()
    dataset.load_poses()
    dataset.load_gray()

    window_size = 5
    min_disp = 16
    num_disp = 112 - min_disp


    cam = dataset.calib.K_cam0

    focal_d = cam[0, 0] # focal distance
    principal_point = [cam[0, 2], cam[1, 2]]

    dist_b = dataset.calib.b_gray

    cam_left = np.concatenate((cam, np.matrix([0, 0, 0]).T), axis=1)
    cam_right = np.concatenate((cam, np.matrix([dist_b * cam[0, 0], 0, 0]).T), axis=1)
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cam, 0,
        cam, 0,
        (1241, 376),
        np.matrix([
            [1.,0.,0.],
            [0.,1.,0.],
            [0.,0.,1.]
        ]),
        np.array([dist_b, 0, 0])
    )

    cur_t = np.array([0.,0.,0.])
    cur_rot = np.matrix([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])

    path = []

    real_poses = np.array(dataset.T_w_cam0)[dataset.frame_range]
    pose_index = 0

    last_sift_left = None
    for pair in dataset.gray:
        left = np.uint8(pair.left * 255.)
        right = np.uint8(pair.right * 255.)

        sift_left = SIFTImage(left)
        sift_right = SIFTImage(right)

        if last_sift_left is not None:
            print('doing math')
            last_pts, now_pts = last_sift_left.correspondences(sift_left)

            # do everything.
            E, mask = cv2.findEssentialMat(
                last_pts,
                now_pts,
                cam,
                cv2.RANSAC,
                0.999,
                1.0
            )
            if E is None or E.shape == (1, 1):
                # dang, no essential matrix found
                raise Exception('No essential matrix found')
            elif E.shape[0] > 3:
                # more than one matrix found, just pick the first
                print('We found multiple essential matrices')
                E = E[0:3, 0:3]

            _, ret_R, ret_t, mask = cv2.recoverPose(
                E,
                last_pts,
                now_pts,
                cam
            )

            cur_t = cur_t + np.dot(cur_rot, ret_t).T
            cur_rot = np.dot(ret_R, cur_rot)

            current_transform = np.concatenate((cur_rot, cur_t.T), axis=1)

            np.set_printoptions(suppress=True)
            print('Det: {}'.format(np.linalg.det(current_transform[:, :3])))

            print(current_transform)

            print(real_poses[pose_index])
            pose_index += 1

            path.append(cur_t.reshape(-1, 3))

        last_sift_left = sift_left
        last_sift_right = sift_right

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    real_path = np.dot(
        np.dot(real_poses, [0,0,0,1]),
        np.matrix([
            [1., 0., 0., 0.],
            [0.,-1., 0., 0.],
            [0., 0.,-1., 0.],
            [0., 0., 0., 1.]
        ])
    )

    np_path = np.array(path).reshape(-1, 3)
    
    if SET_FIRST_POSE:
        first_pose = dataset.T_w_cam0[dataset.frame_range[0]]
        np_path = np.dot(np_path, first_pose[:3, :3])
        np_path += real_path[0, :3]
    
    X = np_path[:, 0]
    Y = np_path[:, 1]
    Z = np_path[:, 2]

    ax.plot_wireframe(X, Y, Z, color='red')

    RX = real_path[:, 0]
    RY = real_path[:, 1]
    RZ = real_path[:, 2]

    # ax.plot_wireframe(RZ, RY, RX)
    ax.plot_wireframe(RX, RY, RZ, color='blue')

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
