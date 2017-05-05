#!/usr/bin/env python

import numpy as np
import cv2
import glob
import argparse
import os

import pykitti # version 0.1.2

import matplotlib.pyplot as plt

from utils.ply_export import write_ply
from utils.draw_sift_matches import draw_matches

DEBUG_SIFT = False

sift = cv2.xfeatures2d.SIFT_create()

class SIFTImage(object):
    """ wrapper for image plus SIFT kp and descriptors """
    def __init__(self, image):
        super(SIFTImage, self).__init__()
        self.image = image
        self.kp, self.des = sift.detectAndCompute(image, None)

        self.good_thresh = 0.7

    def correspondences(self, other):
        # find corresponding points in the input image and the template image
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.des, other.des, k=2)

        # Apply Lowe Ratio Test to the keypoints
        # this should weed out unsure matches
        good_keypoints = []
        for m, n in matches:
            if m.distance < self.good_thresh * n.distance:
                good_keypoints.append(m)

        if DEBUG_SIFT:
            draw_matches(
                self.image, self.kp,
                other.image, other.kp,
                good_keypoints[-50:]
            )
        
        # put keypoints from own image in self_pts
        # transform the keypoint data into arrays for homography check
        # grab precomputed points
        self_pts = np.float32(
            [self.kp[m.queryIdx].pt for m in good_keypoints]
        ).reshape(-1, 2)

        # put corresponding keypoints from other image in other_pts
        other_pts = np.float32(
            [other.kp[m.trainIdx].pt for m in good_keypoints]
        ).reshape(-1, 2)

        return (self_pts, other_pts)

if __name__ == '__main__':
    print('loading')

    dataset = pykitti.odometry(
        '/home/wolf/kitti/dataset',
        '00',
        frame_range=range(0, 300, 3)
    )

    dataset.load_calib()
    dataset.load_timestamps()
    dataset.load_poses()
    dataset.load_gray()

    window_size = 5
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

    initial_position = np.array([0., 0., 0., 1.])
    initial_transform = np.matrix([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])

    current_transform = initial_transform

    path = []

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

        Q = np.float32([
            [1, 0, 0, -calibration_matrix[0, 2]],
            [0,-1, 0,  calibration_matrix[1, 2]], # turn points 180 deg around x-axis,
            [0, 0, 0, -calibration_matrix[0, 0]], # so that y-axis looks up
            [0, 0, 1, 0]
        ])

        points = cv2.reprojectImageTo3D(disp, Q)

        if last_points is not None:
            last_pts, now_pts = last_sift_left.correspondences(sift_left)

            # make pts integers for easy indexing
            last_pts_i = last_pts.astype(np.uint32)
            now_pts_i = now_pts.astype(np.uint32)


            # out_points = last_points[last_pts_i[:, 1], last_pts_i[:, 0]]
            # out_gray = last_sift_left.image[last_pts_i[:, 1], last_pts_i[:, 0]]
            # out_file = 'out-sift-0.ply'
            # write_ply(out_file, out_points, out_gray)


            # out_points = points[now_pts_i[:, 1], now_pts_i[:, 0]]
            # out_gray = left[now_pts_i[:, 1], now_pts_i[:, 0]]
            # out_file = 'out-sift-1.ply'
            # write_ply(out_file, out_points, out_gray)

            # get the 3D coordinates of the matched features
            last_3d_coords = last_points[last_pts_i[:, 1], last_pts_i[:, 0]]
            now_3d_coords = points[now_pts_i[:, 1], now_pts_i[:, 0]]

            retval, out, inliers = cv2.estimateAffine3D(
                last_3d_coords,
                now_3d_coords,
                None,
                None,
                2,

            )

            print('Det: {}'.format(np.linalg.det(out[:, :3])))

            affine_transform = np.concatenate((out, [[0, 0, 0, 1]]))

            current_transform = np.dot(affine_transform, current_transform)

            new_position = np.dot(current_transform, initial_position)

            path.append(new_position)
            print(new_position)


       

        last_points = points
        last_sift_left = sift_left
        last_sift_right = sift_right
        
        # mask = disp > disp.min()
        # out_points = points[mask]
        # out_gray = left[mask]
        # out_file = 'out.ply'
        # write_ply(out_file, out_points, out_gray)


    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    np_path = np.array(path).reshape(-1, 4)

    X = np_path[:, 0]
    Y = np_path[:, 1]
    Z = np_path[:, 2]

    ax.plot_wireframe(X, Y, Z)

    plt.show() 