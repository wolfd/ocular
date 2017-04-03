#!/usr/bin/env python

import numpy as np
import cv2

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ocular.msg import KeypointMotion


class OpticalFlowMatcher(object):
    """
    Optical Flow Matcher for ROS image data
    Reference: http://docs.opencv.org/3.2.0/d7/d8b/tutorial_py_lucas_kanade.html
    """
    def __init__(self, image_topic, feature_detector='FAST'):
        super(OpticalFlowMatcher, self).__init__()

        rospy.init_node('optical_flow_matcher')

        self.cv_bridge = CvBridge()

        self.rectified_image_topic = rospy.Subscriber(
            image_topic,
            Image,
            self.new_image_callback
        )

        self.pub_keypoint_motion = rospy.Publisher(
            'keypoint_motion',
            KeypointMotion,
            queue_size=10
        )

        self.feature_params = None

        if feature_detector == 'FAST':
            self.get_features = self.get_features_fast
            # Initiate FAST detector with default values
            self.fast = cv2.FastFeatureDetector_create()

            self.fast.setThreshold(20)
        elif feature_detector == 'GOOD':
            self.get_features = self.get_features_good
            # params for ShiTomasi 'GOOD' corner detection
            self.feature_params = dict(
                maxCorners=200,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )
        else:
            raise Exception(
                '{} feature detector not implemented'.format(feature_detector)
            )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03
            )
        )

        self.last_frame_gray = None
        self.good_old = None
        self.good_new = None

    def new_image_callback(self, new_image_msg):
        """
        Process new image from ROS
        """
        self.process_new_frame(
            self.cv_bridge.imgmsg_to_cv2(
                new_image_msg,
                desired_encoding="bgr8"
            )
        )

    def process_new_frame(self, new_frame):
        # convert this frame to grayscale
        frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # for first frame, bail fast
        if self.last_frame_gray is None:
            self.store_as_last_frame(frame_gray)
            return

        # use OpenCV to calculate optical flow
        new_frame_matched_features, status, error = cv2.calcOpticalFlowPyrLK(
            self.last_frame_gray,
            frame_gray,
            self.last_frame_features,
            None,
            **self.lk_params
        )

        self.publish_interframe_motion(
            self.last_frame_features,
            new_frame_matched_features,
            status,
            error
        )

        # save data for next frame
        self.store_as_last_frame(frame_gray)

    def store_as_last_frame(self, frame_gray):
        """
        Take a gray frame and store it on the object and get some features from
        it
        """
        self.last_frame_gray = frame_gray
        # we got to make sure that we have features too
        self.last_frame_features = self.get_features(frame_gray)

    def get_features_fast(self, frame_gray):
        """
        Use the FAST feature detection algorithm to find features
        """
        keypoints = self.fast.detect(frame_gray, None)

        return np.float32(
            [kp.pt for kp in keypoints]
        ).reshape(-1, 1, 2)

    def get_features_good(self, frame_gray):
        """
        Jianbo Shi and Carlo Tomasi wrote a paper in 1994 called
        "Good features to track", so now it's called that in OpenCV.
        """
        strong_corners = cv2.goodFeaturesToTrack(
            frame_gray,
            mask=None,
            **self.feature_params
        )

        return strong_corners

    def publish_interframe_motion(self, last_features, new_features, status, err):
        self.good_old = last_features[(status == 1) & (err < 12.0)]
        self.good_new = new_features[(status == 1) & (err < 12.0)]

        # TODO: clean up these features before publishing

        self.pub_keypoint_motion.publish(
            header=Header(
                stamp=rospy.Time.now(),  # TODO: use camera image time
                frame_id='tango_camera_2d'
            ),
            prev_x=self.good_old[:, 0],
            prev_y=self.good_old[:, 1],
            cur_x=self.good_new[:, 0],
            cur_y=self.good_new[:, 1]
        )

    def run(self, debug=False):
        """ The main run loop"""

        # Create some random colors
        color = np.random.randint(0, 255, (10000, 3))

        r = rospy.Rate(100)
        while not rospy.is_shutdown():
            if not debug:
                r.sleep()
                continue

            if self.good_new is None:
                r.sleep()
                continue

            # Create a mask image for drawing purposes
            mask = np.zeros_like(self.last_frame_gray)
            frame = None

            # draw the tracks
            for i, (new, old) in enumerate(zip(self.good_new, self.good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(
                    mask,
                    (a, b),
                    (c, d),
                    color[i].tolist(),
                    2
                )

                frame = cv2.circle(
                    self.last_frame_gray,
                    (a, b),
                    5,
                    color[i].tolist(),
                    -1
                )
            img = cv2.add(frame, mask)
            cv2.imshow('frame', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            r.sleep()

if __name__ == '__main__':
    of_matcher = OpticalFlowMatcher(
        'tango/camera/fisheye_1/image_rect'
    )

    of_matcher.run(debug=True)
