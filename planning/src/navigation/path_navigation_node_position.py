#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import sys
import time
import rospy
import tf
import message_filters

import numpy as np

from nav_msgs.msg import Path
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseStamped, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, Float32, Empty

class PathNavigation:

    def __init__(self):
        self.alpha = rospy.get_param('/planning/interpolation/alpha')
        self.SE2_radius = rospy.get_param('/planning/interpolation/radius')
        self.world_frame_id = rospy.get_param('/octomap/world_frame_id')
        
        self.dist = rospy.get_param('/planning/path_follow_check_radius')
        rospy.Subscriber("/reset_goal_byuser", Empty, self.resetgoatCallBack)

        self.path_sub = rospy.Subscriber('/planner/path', Path, self.path_callback, queue_size = 100)
        
        self.collision_sub = message_filters.Subscriber('/planner/collision', Bool, queue_size = 1)
        self.collision_sub.registerCallback(self.collision_callback)
        
        self.collision = False
        self.is_tracking = False
        self.reset = False

        self.position_cmd_pub = rospy.Publisher('/way_point', PointStamped, queue_size = 1)
        
        self.world_frame_id = 'map'
        self.robot_frame_id = 'vehicle'
        self.tf_listener = tf.TransformListener()
        self.odom = np.array([0.0,0.0,0.0])
        rospy.Subscriber("/state_estimation", Odometry, self.orb_odometryCallBack)

    def get_pose_from_tf(self, from_frame_id):
        # (translation, rotation) = self.tf_listener.lookupTransform(self.world_frame_id,
        #                                                            from_frame_id,
        #                                                            rospy.Time(0))
        # euler = tf.transformations.euler_from_quaternion(rotation)
        # return np.array([translation[0], translation[1], euler[2]])
        return np.array([self.odom[0], self.odom[1], 0.0])

    def orb_odometryCallBack(self, msg):
        self.odom[0] = msg.pose.pose.position.x
        self.odom[1] = msg.pose.pose.position.y

    def path_callback(self, path_msg):
        rospy.loginfo("Received a path from exploration algorithm!")
        if self.is_tracking:
            self.collision = True
        
        traj = [path_msg.poses[0].pose]
        
        for path_waypoint in path_msg.poses[1:]:

            traj.append(path_waypoint.pose)
        
        self.publish_traj(traj)

    def collision_callback(self, collision_msg):
	      self.collision = collision_msg.data

    def publish_traj(self, traj):
        self.is_tracking = True
        self.is_tracking = True
        self.collision = False
        # We select certain points along the path
        n_points = int(len(traj))
        if n_points <3:
            new_nodes = np.arange(0,n_points)
        elif n_points <6:
            plan_nodes = np.ceil(n_points / 2)
            new_nodes = np.sort(np.random.choice(np.arange(1, n_points - 2), int(plan_nodes), replace=False))
            new_nodes = np.append(new_nodes, n_points-1)
        else:
            plan_nodes = np.ceil(n_points / 5)
            new_nodes = np.sort(np.random.choice(np.arange(1, n_points - 2), int(plan_nodes), replace=False))
            new_nodes = np.append(new_nodes, n_points-1)
        # for pose in traj:
        for i in new_nodes:
            pose = traj[i]
            while True:
                if self.collision:
                    print("collision!")
                    break
                robot_pose = self.get_pose_from_tf(self.robot_frame_id)
                # print("robot pose:" + str(robot_pose[0]) + "," + str(robot_pose[1]) + "," + str(robot_pose[2]))
                dist = np.sqrt((robot_pose[0] - pose.position.x)**2 + (robot_pose[1] - pose.position.y)**2)
                if dist < self.dist:
                    break

                if self.reset:
                    self.reset = False
                    return

                position_cmd_msg = PoseStamped()
                position_cmd_msg.pose = pose
                position_cmd_msg.header.frame_id = self.world_frame_id
                position_cmd_msg.header.stamp = rospy.Time.now()

                pose_cmd = PointStamped()
                pose_cmd.header.frame_id = self.world_frame_id
                pose_cmd.header.stamp = rospy.Time.now()
                pose_cmd.point.x = pose.position.x
                pose_cmd.point.y = pose.position.y

                self.position_cmd_pub.publish(pose_cmd)
                rospy.sleep(3)

        self.is_tracking = False
    def resetgoatCallBack(self, msg):
        self.reset = True
        rospy.loginfo("Reset goal by user!")

    def interpolate(self, pose_1, pose_2):
        xi_1 = self.get_se2_from_pose_msg(pose_1)
        xi_2 = self.get_se2_from_pose_msg(pose_2)
        
        T_1 = self.se2_to_SE2(xi_1)
        T_2 = self.se2_to_SE2(xi_2)
        
        T_12 = np.matmul(np.linalg.inv(T_1), T_2)
        xi_12 = self.SE2_to_se2(T_12)
        T = np.matmul(T_1, self.se2_to_SE2(self.alpha * xi_12))
        xi = self.SE2_to_se2(T)
        
        interploted_pose = Pose()
        interploted_pose.position.x = xi[0]
        interploted_pose.position.y = xi[1]
        interploted_pose.position.z = 0
        quaternion = tf.transformations.quaternion_from_euler(0, 0, xi[2])
        interploted_pose.orientation.x = quaternion[0]
        interploted_pose.orientation.y = quaternion[1]
        interploted_pose.orientation.z = quaternion[2]
        interploted_pose.orientation.w = quaternion[3]
        
        return interploted_pose
        
    def close_enough(self, pose_1, pose_2):
        xi_1 = self.get_se2_from_pose_msg(pose_1)
        xi_2 = self.get_se2_from_pose_msg(pose_2)
        
        T_1 = self.se2_to_SE2(xi_1)
        T_2 = self.se2_to_SE2(xi_2)
        
        T_12 = np.matmul(np.linalg.inv(T_1), T_2)
        xi_12 = self.SE2_to_se2(T_12)
        
        distance = np.linalg.norm(xi_12)
        
        if distance < self.SE2_radius:
            return True
        else:
            return False
        
    @staticmethod
    def get_se2_from_pose_msg(pose_msg):
        x = pose_msg.position.x
        y = pose_msg.position.y
        theta = tf.transformations.euler_from_quaternion([pose_msg.orientation.x,
                                                          pose_msg.orientation.y,
                                                          pose_msg.orientation.z,
                                                          pose_msg.orientation.w])[2]
        return np.array([x, y, theta])
        
    @staticmethod
    def se2_to_SE2(xi):
        sin = np.sin(xi[2])
        cos = np.cos(xi[2])
        
        if xi[2] != 0:
            V = 1 / xi[2] * np.array([[sin, cos - 1], [1 - cos, sin]])
        else:
            V = np.eye(2)
        
        Vu = np.matmul(V, xi[:2, None]).squeeze()
        
        T = np.array([[cos, -sin, Vu[0]],
                      [sin,  cos, Vu[1]],
                      [0,    0,   1]])
        
        return T
    
    @staticmethod
    def SE2_to_se2(T):
        theta = np.arctan2(T[1, 0], T[0, 0])
        if theta != 0:
            A = T[1, 0] / theta
            B = (1 - T[0, 0]) / theta
            V_inv = np.array([[A, B], [-B, A]]) / (A**2 + B**2)
        else:
            V_inv = np.eye(2)
        
        u = np.matmul(V_inv, T[:2, 2])
        
        xi = np.array([u[0], u[1], theta])
        
        return xi
            

def main(args):
    rospy.init_node('path_navigation', anonymous=True)
    path_navigation = PathNavigation()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Navigation stopped!")


if __name__ == '__main__':
    main(sys.argv)
