#!/usr/bin/env python

import numpy as np
import cv2

import rospy

from ocular.msg import KeypointMotion


class InterframeCameraEstimator(object):
    """
    Takes interframe keypoint motion and generates a camera matrix from it
    Uses fundamental & essential matrices (assuming pre-calibrated camera from
    Tango). Then does SVD to get some matrices and figures out which one makes
    sense.

    Publishes the estimated pose delta to 'raw_visual_pose_delta' topic.
    """
    def __init__(self):
        super(InterframeCameraEstimator, self).__init__()

        rospy.init_node('interframe_camera_estimator')

        self.interframe_keypoint_motion_topic = rospy.Subscriber(
            'keypoint_motion',
            KeypointMotion,
            self.new_motion_callback
        )

    def new_motion_callback(self, new_motion_msg):
        """
        Take keypoint motion data from other node and process it
        """
        previous_kp = np.stack(
            (new_motion_msg.prev_x, new_motion_msg.prev_y),
            axis=0
        )

        current_kp = np.stack(
            (new_motion_msg.cur_x, new_motion_msg.cur_y),
            axis=0
        )

        print previous_kp.shape
        print current_kp.shape

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(100)
        while not rospy.is_shutdown():
            r.sleep()


if __name__ == '__main__':
    cam_estimator = InterframeCameraEstimator()

    cam_estimator.run()
