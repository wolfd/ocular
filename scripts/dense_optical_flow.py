#!/usr/bin/env python

import cv2
import numpy as np


class DenseOpticalFlow(object):
    def __init__(self):
        super(DenseOpticalFlow, self).__init__()

    def calculate_flow(self, frame_a, frame_b):
        previous_frame = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            previous_frame,
            next_frame,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Change here
        horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        horz = horz.astype('uint8')
        vert = vert.astype('uint8')

        # Change here too
        cv2.imshow('Horizontal Component', horz)
        cv2.imshow('Vertical Component', vert)

        k = cv2.waitKey(0) & 0xff
        if k == ord('s'):  # Change here
            cv2.imwrite('opticalflow_horz.pgm', horz)
            cv2.imwrite('opticalflow_vert.pgm', vert)

        cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    of = DenseOpticalFlow()
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_file)

    ret, frame_last = cap.read()
    frame_next = None

    while True:
        ret, frame_next = cap.read()
        of.calculate_flow(
            frame_last,
            frame_next
        )

        frame_last = frame_next
