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

if __name__ == '__main__':
    print('loading')

    dataset = pykitti.odometry(
        '/home/wolf/kitti/dataset',
        '00',
        frame_range=range(0, 500, 2)
    )

    dataset.load_calib()
    dataset.load_timestamps()
    dataset.load_poses()
    dataset.load_gray()

    window_size = 5
    min_disp = 16
    num_disp = 112 - min_disp


    cam = dataset.calib.K_cam0
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

    # alternative parameters
    # window_size = 3
    # min_disp = 16
    # num_disp = 128

    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=min_disp,
    #     numDisparities=num_disp,
    #     blockSize=16,
    #     P1=4*window_size**2,
    #     P2=32*window_size**2,
    #     disp12MaxDiff=1,
    #     uniquenessRatio=10,
    #     speckleWindowSize=100,
    #     speckleRange=32
    # )


    initial_position = np.array([0., 0., 0., 1.])
    initial_transform = np.matrix([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])

    current_transform = initial_transform

    path = []

    real_poses = np.array(dataset.T_w_cam0)[dataset.frame_range]
    pose_index = 0

    last_points = None
    for pair in dataset.gray:
        print('computing disparity')

        left = np.uint8(pair.left * 255.)
        right = np.uint8(pair.right * 255.)

        sift_left = SIFTImage(left)
        sift_right = SIFTImage(right)

        disp = stereo.compute(
            left,
            right
        ).astype(np.float32) / 16.0

        calibration_matrix = dataset.calib.K_cam0 # left gray camera (they're all the same here)

        # Q = np.float32([
        #     [1, 0, 0, -calibration_matrix[0, 2]],
        #     [0,-1, 0,  calibration_matrix[1, 2]], # turn points 180 deg around x-axis,
        #     [0, 0, 0, -calibration_matrix[0, 0]], # so that y-axis looks up
        #     [0, 0, 1, 0]
        # ])

        points = cv2.reprojectImageTo3D(disp, Q)

        if last_points is not None:
            last_pts, now_pts = last_sift_left.correspondences(sift_left)

            # make pts integers for easy indexing
            last_pts_i = last_pts.astype(np.uint32)
            now_pts_i = now_pts.astype(np.uint32)

            if DEBUG_SIFT_3D:
                out_points = last_points[last_pts_i[:, 1], last_pts_i[:, 0]]
                out_gray = last_sift_left.image[last_pts_i[:, 1], last_pts_i[:, 0]]
                out_file = 'out-sift-0.ply'
                write_ply(out_file, out_points, out_gray)


                out_points = points[now_pts_i[:, 1], now_pts_i[:, 0]]
                out_gray = left[now_pts_i[:, 1], now_pts_i[:, 0]]
                out_file = 'out-sift-1.ply'
                write_ply(out_file, out_points, out_gray)

            # get the 3D coordinates of the matched features
            last_3d_coords = last_points[last_pts_i[:, 1], last_pts_i[:, 0]]
            now_3d_coords = points[now_pts_i[:, 1], now_pts_i[:, 0]]
            ret_R, ret_t = rigid_transform_3D(last_3d_coords[-50:], now_3d_coords[-50:])

            out = np.concatenate((ret_R, np.matrix(ret_t).T), axis=1)

            # retval, out, inliers = cv2.estimateAffine3D(
            #     last_3d_coords,
            #     now_3d_coords,
            #     None,
            #     None,
            #     2
            # )

            print('Det: {}'.format(np.linalg.det(out[:, :3])))

            affine_transform = np.concatenate((out, [[0, 0, 0, 1]]))
            np.set_printoptions(suppress=True)
            # print(affine_transform)

            current_transform = np.dot(affine_transform, current_transform)
            print(current_transform)

            print(real_poses[pose_index])
            pose_index += 1

            new_position = np.dot(current_transform, initial_position)

            path.append(new_position)
            # print(new_position)


       

        last_points = points
        last_sift_left = sift_left
        last_sift_right = sift_right
        
        if DEBUG_STEREO_3D:
            mask = disp > disp.min()
            out_points = points[mask]
            out_gray = left[mask]
            out_file = 'out.ply'
            write_ply(out_file, out_points, out_gray)


    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    np_path = np.array(path).reshape(-1, 4)

    X = np_path[:, 0]
    Y = np_path[:, 1]
    Z = np_path[:, 2]

    real_path = np.dot(real_poses, [0,0,0,1])

    ax.plot_wireframe(X, Y, Z)

    RX = real_poses[:, 0]
    RY = real_poses[:, 1]
    RZ = real_poses[:, 2]

    ax.plot_wireframe(RZ, RY, RX)

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
