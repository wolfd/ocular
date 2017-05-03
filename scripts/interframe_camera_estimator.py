#!/usr/bin/env python

import numpy as np
import cv2

import rospy
import tf

from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, PointCloud
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from geometry_msgs.msg import Point32

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

        self.pub_point_cloud = rospy.Publisher(
            'interframe_point_cloud',
            PointCloud,
            queue_size=10
        )

        self.camera_intrinsics = None

        self.accumulated_pose = make_empty_pose()

        self.base_transformation_mat = np.matrix([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ], np.float32)

    def new_camera_intrinsics_callback(self, new_camera_info):
        """
        Store the camera intrinsics.
        We need this for the calibration matrices from the Tango
        """
        self.camera_intrinsics = new_camera_info
        self.k_mat = np.matrix(
            np.array(self.camera_intrinsics.K).reshape((3, 3))
        )

        self.k_inv = self.k_mat.I

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

        error_amount, triangulated = self.triangulation(
            previous_kp, current_kp,
            self.base_transformation_mat, camera_matrix
        )

        # print np.linalg.norm(np.array(error_amount))
        for p in triangulated:
            print p

        self.pub_point_cloud.publish(
            header=Header(
                stamp=rospy.Time.now(),  # TODO: use camera image time
                frame_id='map'
            ),
            points=[Point32(p[0], p[1], p[2]) for p in triangulated]
        )

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

    def triangulation(self, kp_a, kp_b, cam_a, cam_b):
        """
        Returns a point cloud
        """
        reproj_error = []
        point_cloud = []
        for i in range(len(kp_a)):
            # convert to normalized homogeneous coordinates
            kp = kp_a[i]
            u = np.array([kp[0], kp[1], 1.0])

            mat_um = self.k_inv * np.matrix(u).T

            u = np.array(mat_um[:, 0])

            kp_ = kp_b[i]
            u_ = np.array([kp_[0], kp_[1], 1.0])

            mat_um_ = self.k_inv * np.matrix(u_).T

            u_ = np.array(mat_um_[:, 0])

            # now we triangulate!
            x = self.linear_ls_triangulation(
                u, cam_a, u_, cam_b
            )

            point_cloud.append(x.flatten())

            # calculate reprojection error

            # reproject to other img
            x_for_camera = np.matrix(
                np.append(x, [[1.0]], axis=0)
            )

            x_pt_img = np.array(self.k_mat * cam_b * x_for_camera).flatten()
            x_pt_img_ = np.array([
                x_pt_img[0] / x_pt_img[2],
                x_pt_img[1] / x_pt_img[2]
            ])

            # check error in matched keypoint
            reproj_error.append(
                np.linalg.norm(x_pt_img_ - kp_)
            )

        return reproj_error, point_cloud

    def linear_ls_triangulation(self, point_a, cam_a, point_b, cam_b):
        """
        Python version of
        Mastering Opencv With Practical Computer Vision Projects'
        LST implementation on page 144
        """
        # build A matrix
        # import pdb; pdb.set_trace()

        point_a = point_a.flatten()
        point_b = point_b.flatten()

        mat_a = np.matrix([
            [point_a[0]*cam_a[2, 0]-cam_a[0, 0], point_a[0]*cam_a[2, 1]-cam_a[0, 1], point_a[0]*cam_a[2, 2]-cam_a[0, 2]],
            [point_a[1]*cam_a[2, 0]-cam_a[1, 0], point_a[1]*cam_a[2, 1]-cam_a[1, 1], point_a[1]*cam_a[2, 2]-cam_a[1, 2]],
            [point_b[0]*cam_b[2, 0]-cam_b[0, 0], point_b[0]*cam_b[2, 1]-cam_b[0, 1], point_b[0]*cam_b[2, 2]-cam_b[0, 2]],
            [point_b[1]*cam_b[2, 0]-cam_b[1, 0], point_b[1]*cam_b[2, 1]-cam_b[1, 1], point_b[1]*cam_b[2, 2]-cam_b[1, 2]]
        ])
        # build B vector
        mat_b = np.matrix([
            [-(point_a[0]*cam_a[2, 3]-cam_a[0, 3])],
            [-(point_a[1]*cam_a[2, 3]-cam_a[1, 3])],
            [-(point_b[0]*cam_b[2, 3]-cam_b[0, 3])],
            [-(point_b[1]*cam_b[2, 3]-cam_b[1, 3])]
        ])

        # solve for X
        _, x = cv2.solve(mat_a, mat_b, None, cv2.DECOMP_SVD)

        return x

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
