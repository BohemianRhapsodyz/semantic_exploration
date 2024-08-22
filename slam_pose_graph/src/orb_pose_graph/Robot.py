#!/usr/bin/env python3

# jplaced
# jjgomez
# 2022, Universidad de Zaragoza
# Modified by Rongge Zhang
# 2024 Polytechnique Montreal


import rospy
import tf
# import actionlib
import numpy as np

from scipy.spatial.transform import Rotation
from typing import Tuple

# from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
# from nav_msgs.srv import GetPlan
from geometry_msgs.msg import Pose, PoseStamped

from orb_pose_graph.Functions import yawBtw2Points
from nav_msgs.msg import Odometry

class Robot:
    def __init__(self, name: str):
        """
        Constructor
        """
        self.start = PoseStamped()
        self.end = PoseStamped()
        self.pose = Pose()

        self.assigned_point = []
        self.name = name  # robot_1
        rospy.loginfo(rospy.get_name() + ': Robot Class started with robot name: ' + name)
        self.odom = np.array([0.0,0.0,0.0])
        self.quat = np.array([0.0, 0.0, 0.0, 0.0])
        self.global_frame = 'map'
        self.robot_frame = 'vehicle'
        self.listener = tf.TransformListener()
        rospy.Subscriber("/state_estimation", Odometry, self.orb_odometryCallBack)
        cond = 0
        while cond == 0:
            try:
                rospy.loginfo(rospy.get_name() + ': Robot Class is waiting for the robot transform.')
                (trans, rot) = self.listener.lookupTransform(self.global_frame, self.robot_frame, rospy.Time(0))
                self.position = np.array([trans[0], trans[1]])
                self.rotation = np.array([rot[0], rot[1], rot[2], rot[3]])
                self.pose.position.x = trans[0]
                self.pose.position.y = trans[1]
                self.pose.position.z = 0
                self.pose.orientation.x = rot[0]
                self.pose.orientation.y = rot[1]
                self.pose.orientation.z = rot[2]
                self.pose.orientation.w = rot[3]
                cond = 1
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                cond = 0
        rospy.loginfo(rospy.get_name() + ': Robot Class received the robot transform.')

        self.assigned_point = self.position

        self.start.header.frame_id = self.global_frame
        self.end.header.frame_id = self.global_frame
        rospy.loginfo(rospy.get_name() + ': Initialized robot.')

    def orb_odometryCallBack(self, msg):
        self.odom[0] = msg.pose.pose.position.x
        self.odom[1] = msg.pose.pose.position.y

        self.quat[0] = msg.pose.pose.orientation.x
        self.quat[1] = msg.pose.pose.orientation.y
        self.quat[2] = msg.pose.pose.orientation.z
        self.quat[3] = msg.pose.pose.orientation.w
    def getPosition(self) -> np.array:
        """
        Gets robot's current position
        """
        self.position = np.array(([self.odom[0], self.odom[1]]))
        return self.position

    def getPose(self) -> Tuple[np.array, np.array]:
        """
        Gets robot's current pose as numpy arrays
        """
        self.position = np.array(([self.odom[0], self.odom[1]]))
        self.rotation = np.array(([self.quat[0],self.quat[1],self.quat[2],self.quat[3]]))
        return self.position, self.rotation

    def getPoseAsGeometryMsg(self) -> Pose:
        """
        Gets robot's current pose as geometry msg
        """
        self.pose.position.x = self.odom[0]
        self.pose.position.y = self.odom[1]
        self.pose.position.z = 0
        self.pose.orientation.x = self.quat[0]
        self.pose.orientation.y = self.quat[1]
        self.pose.orientation.z = self.quat[2]
        self.pose.orientation.w = self.quat[3]

        return self.pose
