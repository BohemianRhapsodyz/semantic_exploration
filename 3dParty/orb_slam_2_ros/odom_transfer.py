import rospy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
import math
from pyquaternion import Quaternion
import tf
import sys
import numpy as np

# vehicle_type = sys.argv[1]
# vehicle_id = sys.argv[2]
local_pose = Odometry()
local_pose.header.frame_id = 'map'
quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)
q = Quaternion([quaternion[3],quaternion[0],quaternion[1],quaternion[2]])

def vins_callback(data):
    local_pose.pose.pose.position.x = data.pose.position.x
    local_pose.pose.pose.position.y = data.pose.position.y
    local_pose.pose.pose.position.z = 0.5
    q_= Quaternion([data.pose.orientation.w,data.pose.orientation.x,data.pose.orientation.y,data.pose.orientation.z])
    q_ = q_*q
    local_pose.pose.pose.orientation.w = q_[0]
    local_pose.pose.pose.orientation.x = q_[1]
    local_pose.pose.pose.orientation.y = q_[2]
    local_pose.pose.pose.orientation.z = q_[3]


rospy.init_node('odom_transfer')
rospy.Subscriber("orb_slam2_rgbd/camera_pose", PoseStamped, vins_callback,queue_size=1)
position_pub = rospy.Publisher("/state_estimation", Odometry, queue_size=1)
rate = rospy.Rate(100)

while not rospy.is_shutdown():
    if (local_pose.pose.pose.position == Point()):
        continue
    else:
        print("Pose received")
        local_pose.header.stamp = rospy.Time.now()
        position_pub.publish(local_pose)
    try:
        rate.sleep()
    except:
        continue
