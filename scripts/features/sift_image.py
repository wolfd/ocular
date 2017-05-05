#!/usr/bin/env python

import numpy as np
import cv2

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
