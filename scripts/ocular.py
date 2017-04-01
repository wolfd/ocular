#!/usr/bin/env python

import rospy
import tf
import time
from std_msgs.msg import String, Header
from geometry_msgs.msg import Quaternion, Vector3
from sensor_msgs.msg import Imu, PointCloud2, LaserScan, Image, CameraInfo
from dynamic_reconfigure.server import Server
#from ocular.cfg import ocularconfConfig as dynserv

rospy.init_node('ocular')


class ocular(object):
	def __init__(self):
		serv = Server(dynserv, self.config_callback)


	def config_callback(self, config, level):
		self.scan_frame = config.scan_frame