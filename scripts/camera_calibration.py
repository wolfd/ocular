#!/usr/bin/env python

import numpy as np
import cv2
import glob
import argparse
import os

CHESSBOARD_COLUMNS = 10
CHESSBOARD_ROWS = 7

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESSBOARD_ROWS*CHESSBOARD_COLUMNS, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_COLUMNS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

parser = argparse.ArgumentParser()
parser.add_argument("images_directory")
args = parser.parse_args()

images = glob.glob(os.path.join(args.images_directory, '*.jpg'))

for fname in images:
    print('reading image {}'.format(fname))
    img = cv2.imread(fname)
    print('finished reading image {}'.format(fname))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('converted image {} to grayscale'.format(fname))

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_COLUMNS, CHESSBOARD_ROWS), None)
    print('found chessboard corner {}'.format(fname))
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (CHESSBOARD_COLUMNS, CHESSBOARD_ROWS), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()
