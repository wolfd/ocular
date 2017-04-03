#!/usr/bin/env python

import numpy as np
import cv2

import rospy
import tf

from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

from ocular.msg import KeypointMotion


def make_empty_pose():
    return Pose(
        Point(0, 0, 0),
        Quaternion(0, 0, 0, 1)
    )


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

        self.camera_intrinsics_topic = rospy.Subscriber(
            'tango/camera/fisheye_1/camera_info',
            CameraInfo,
            self.new_camera_intrinsics_callback
        )

        self.pub_pose = rospy.Publisher(
            'interframe_pose',
            PoseStamped,
            queue_size=10
        )

        self.camera_intrinsics = None

        self.accumulated_pose = make_empty_pose()

    def new_camera_intrinsics_callback(self, new_camera_info):
        """
        Store the camera intrinsics.
        We need this for the calibration matrices from the Tango
        """
        self.camera_intrinsics = new_camera_info
        self.k_mat = np.matrix(
            np.array(self.camera_intrinsics.K).reshape((3, 3))
        )

    def new_motion_callback(self, new_motion_msg):
        """
        Take keypoint motion data from other node and process it
        """

        # we can't do anything until we have the camera calibration
        if self.camera_intrinsics is None:
            # TOmaybeDO: use a wait_for_message instead of missing a frame?
            return

        previous_kp = np.stack(
            (new_motion_msg.prev_x, new_motion_msg.prev_y),
            axis=1
        )

        current_kp = np.stack(
            (new_motion_msg.cur_x, new_motion_msg.cur_y),
            axis=1
        )

        f_mat = self.calculate_fundamental_matrix(previous_kp, current_kp)

        camera_matrix, R_mat, t_mat = self.manually_calculate_pose(f_mat)

        # get quaternion from rotation matrix
        tf_rot = np.identity(4)
        tf_rot[0:3, 0:3] = R_mat

        quat = tf.transformations.quaternion_from_matrix(tf_rot)

        old_quat = self.accumulated_pose.orientation

        new_quat = tf.transformations.quaternion_multiply(
            [old_quat.x, old_quat.y, old_quat.z, old_quat.w],
            quat
        )

        normalized_new_quat = tf.transformations.quaternion_from_euler(
            *tf.transformations.euler_from_quaternion(new_quat)
        )

        print normalized_new_quat

        self.accumulated_pose.orientation = Quaternion(
            *normalized_new_quat
        )

        self.pub_pose.publish(
            header=Header(
                stamp=rospy.Time.now(),  # TODO: use camera image time
                frame_id='map'
            ),
            pose=Pose(
                Point(
                    0, 0, 0
                ),
                self.accumulated_pose.orientation
            )
        )

        # import pdb; pdb.set_trace()

    def calculate_fundamental_matrix(self, previous_pts, current_pts):
        fundamental_matrix, mask = cv2.findFundamentalMat(
            previous_pts,
            current_pts,
            cv2.FM_RANSAC
        )

        if fundamental_matrix is None or fundamental_matrix.shape == (1, 1):
            # dang, no fundamental matrix found
            raise Exception('No fundamental matrix found')
        elif fundamental_matrix.shape[0] > 3:
            # more than one matrix found, just pick the first
            fundamental_matrix = fundamental_matrix[0:3, 0:3]

        return np.matrix(fundamental_matrix)

    def manually_calculate_pose(self, f_mat):
        # get essential matrix from the fundamental
        # I am assuming that only one calibration matrix is fine here, because
        # only one type of camera is being used.
        e_mat = self.k_mat.T * f_mat * self.k_mat

        singular_values, u_mat, vt = cv2.SVDecomp(e_mat)
        # reconstruction from SVD:
        # np.dot(u_mat, np.dot(np.diag(singular_values.T[0]), vt))

        u_mat = np.matrix(u_mat)
        vt = np.matrix(vt)

        # from Epipolar Geometry and the Fundamental Matrix 9.13
        w_mat = np.matrix([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], np.float32)

        R_mat = u_mat * w_mat * vt  # HZ 9.19
        t_mat = u_mat[:, 2]  # get third column of u

        # check rotation matrix for validity
        if np.linalg.det(R_mat) - 1.0 > 1e-07:
            print('{}\nDoes not appear to be a valid rotation matrix'.format(
                R_mat
            ))

        camera_matrix = np.column_stack((R_mat, t_mat))

        return camera_matrix, R_mat, t_mat

    def run(self):
        """ The main run loop"""
        r = rospy.Rate(100)
        while not rospy.is_shutdown():
            r.sleep()


if __name__ == '__main__':
    cam_estimator = InterframeCameraEstimator()

    cam_estimator.run()
