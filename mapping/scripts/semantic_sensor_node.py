#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

import sys
import rospy
import numpy as np
import cv2
import message_filters
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from skimage.transform import resize
from sensor_msgs.msg import PointCloud2, PointField
from enum import Enum

class PointType(Enum):
    SEMANTIC = 0

class SemanticPclGenerator:
    def __init__(self, intrinsic, width = 80, height = 60, frame_id = "/camera",
                 point_type = PointType.SEMANTIC):
        '''
        width: (int) width of input images
        height: (int) height of input images
        '''
        self.point_type = point_type
        self.intrinsic = intrinsic
        # Allocate arrays
        x_index = np.array([list(range(width))*height], dtype = '<f4')
        y_index = np.array([[i]*width for i in range(height)], dtype = '<f4').ravel()
        self.xy_index = np.vstack((x_index, y_index)).T # x,y
        self.xyd_vect = np.zeros([width*height, 3], dtype = '<f4') # x,y,depth
        self.XYZ_vect = np.zeros([width*height, 3], dtype = '<f4') # real world coord
        self.ros_data = np.ones([width*height, 6], dtype = '<f4') # [x,y,z,0,bgr0,semantic_color]
        self.bgr0_vect = np.zeros([width*height, 4], dtype = '<u1') #bgr0
        self.semantic_color_vect = np.zeros([width*height, 4], dtype = '<u1') #bgr0
        # Prepare ros cloud msg
        # Cloud data is serialized into a contiguous buffer, set fields to specify offsets in buffer
        self.cloud_ros = PointCloud2()
        self.cloud_ros.header.frame_id = frame_id
        self.cloud_ros.height = 1
        self.cloud_ros.width = width*height
        self.cloud_ros.fields.append(PointField(
            name = "x",
            offset = 0,
            datatype = PointField.FLOAT32, count = 1))
        self.cloud_ros.fields.append(PointField(
            name = "y",
            offset = 4,
            datatype = PointField.FLOAT32, count = 1))
        self.cloud_ros.fields.append(PointField(
            name = "z",
            offset = 8,
            datatype = PointField.FLOAT32, count = 1))
        self.cloud_ros.fields.append(PointField(
            name = "rgb",
            offset = 16,
            datatype = PointField.FLOAT32, count = 1))
        self.cloud_ros.fields.append(PointField(
            name = "semantic_color",
            offset = 20,
            datatype = PointField.FLOAT32, count = 1))
        self.cloud_ros.is_bigendian = False
        self.cloud_ros.point_step = 6 * 4 # In bytes
        self.cloud_ros.row_step = self.cloud_ros.point_step * self.cloud_ros.width * self.cloud_ros.height
        self.cloud_ros.is_dense = False

    def generate_cloud_data_common(self, bgr_img, depth_img):
        """
        Do depth registration, suppose that rgb_img and depth_img has the same intrinsic
        \param bgr_img (numpy array bgr8)
        \param depth_img (numpy array float32 2d)
        [x, y, Z] = [X, Y, Z] * intrinsic.T
        """
        np.place(depth_img, depth_img == 0, 100000) # Handle maximum range measurements

        bgr_img = bgr_img.view('<u1')
        depth_img = depth_img.view('<f4')
        # Add depth information
        self.xyd_vect[:,0:2] = self.xy_index * depth_img.reshape(-1,1)*10/255  # Division by 1000 for unit conversion
        self.xyd_vect[:,2:3] = depth_img.reshape(-1,1)*10/255  # Division by 1000 for unit conversion
        self.XYZ_vect = self.xyd_vect.dot(self.intrinsic.I.T)
        # Convert to ROS point cloud message in a vectorialized manner
        # ros msg data: [x,y,z,0,bgr0,semantic_color] (little endian float32)
        # Transform color
        self.bgr0_vect[:,0:1] = bgr_img[:,:,0].reshape(-1,1)
        self.bgr0_vect[:,1:2] = bgr_img[:,:,1].reshape(-1,1)
        self.bgr0_vect[:,2:3] = bgr_img[:,:,2].reshape(-1,1)
        # Concatenate data
        self.ros_data[:,0:3] = self.XYZ_vect
        self.ros_data[:,4:5] = self.bgr0_vect.view('<f4')

    def make_ros_cloud(self, stamp):
        # Assign data to ros msg
        # We should send directly in bytes, send in as a list is too slow, numpy tobytes is too slow, takes 0.3s.
        self.cloud_ros.data = self.ros_data.ravel().tobytes()
        #self.cloud_ros.header.stamp = stamp
        self.cloud_ros.header.stamp = rospy.Time.now()
        return self.cloud_ros

    def generate_cloud_semantic(self, bgr_img, semantic_color, depth_img, stamp):
        self.generate_cloud_data_common(bgr_img, depth_img)
        #Transform semantic color
        self.semantic_color_vect[:,0:1] = semantic_color[:,:,0].reshape(-1,1)
        self.semantic_color_vect[:,1:2] = semantic_color[:,:,1].reshape(-1,1)
        self.semantic_color_vect[:,2:3] = semantic_color[:,:,2].reshape(-1,1)
        # Concatenate data
        self.ros_data[:,5:6] = self.semantic_color_vect.view('<f4')
        return self.make_ros_cloud(stamp)

class SemanticCloud:
    """
    Class for ros node to take in a color image (bgr) and a semantic segmentation image (bgr)
    Then produce point cloud based on depth information
    """
    def __init__(self):
        # Get point type
        point_type = rospy.get_param('/semantic_pcl/point_type')
        if point_type == 0:
            self.point_type = PointType.SEMANTIC
            print('Generate semantic point cloud.')
        else:
            print("Invalid point type.")
            return
        # Get image size
        self.img_width, self.img_height = rospy.get_param('/camera/width'), rospy.get_param('/camera/height')
        # Set up ROS
        self.bridge = CvBridge() # CvBridge to transform ROS Image message to OpenCV image
        # Set up ros image subscriber
        # Set buff_size to average msg size to avoid accumulating delay
        # Point cloud frame id
        frame_id = rospy.get_param('/semantic_pcl/frame_id')
        # Camera intrinsic matrix
        fx = rospy.get_param('/camera/fx')
        fy = rospy.get_param('/camera/fy')
        cx = rospy.get_param('/camera/cx')
        cy = rospy.get_param('/camera/cy')
        intrinsic = np.matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype = np.float32)
        
        # Noise configuration
        self.noisy_obs = False
        self.depth_noise_std = rospy.get_param('/semantic_pcl/depth_noise_std')
        self.true_class_prob = rospy.get_param('/semantic_pcl/true_class_prob')
        if self.depth_noise_std > 0 or self.true_class_prob < 1:
            self.noisy_obs = True
            num_classes = rospy.get_param('/class_labels/num_classes')
            self.class_colors = []
            for i in range(num_classes):
                color = rospy.get_param('/class_labels/color_' + str(i + 1))
                self.class_colors.append(255 * np.array([color['b'], color['g'], color['r']]))
            self.class_colors = np.array(self.class_colors).astype(np.uint8)
        
        self.pcl_pub = rospy.Publisher("/semantic_pcl/semantic_pcl", PointCloud2, queue_size = 1)

        # increase buffer size to avoid delay (despite queue_size = 1)
        self.color_sub = message_filters.Subscriber(rospy.get_param('/semantic_pcl/color_image_topic'), Image,
                                                    queue_size = 1, buff_size = 10*self.img_width*self.img_height)
        self.semantic_sub = message_filters.Subscriber(rospy.get_param('/semantic_pcl/semantic_image_topic'), Image,
                                                       queue_size = 1, buff_size = 10*self.img_width*self.img_height)
        self.depth_sub = message_filters.Subscriber(rospy.get_param('/semantic_pcl/depth_image_topic'), Image,
                                                    queue_size = 1, buff_size = 10*self.img_width*self.img_height)

        # Take in color image, semantic image, and depth image with a limited time gap between message time stamps
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.semantic_sub, self.depth_sub],
                                                              queue_size = 1, slop = 0.3)
        self.ts.registerCallback(self.color_semantic_depth_callback)
        self.cloud_generator = SemanticPclGenerator(intrinsic, self.img_width,self.img_height, frame_id,
                                                    self.point_type)
        print('Semantic point cloud ready!')

    def color_semantic_depth_callback(self, color_img_ros, semantic_img_ros , depth_img_ros):
        """
        Callback function to produce point cloud registered with semantic class color based
        on input color image and depth image
        """
        # Convert ros Image message to numpy array
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_img_ros, "bgr8")
            semantic_img = self.bridge.imgmsg_to_cv2(semantic_img_ros, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_img_ros, "32FC1")
        except CvBridgeError as e:
            print(e)

        # Resize depth
        if depth_img.shape[0] is not self.img_height or depth_img.shape[1] is not self.img_width:
            depth_img = resize(depth_img, (self.img_height, self.img_width), order = 0, mode = 'reflect',
                               anti_aliasing=False, preserve_range = True) # order = 0, nearest neighbour
            depth_img = depth_img.astype(np.float32)

        # Resize semantic
        if semantic_img.shape[0] is not self.img_height or semantic_img.shape[1] is not self.img_width:
            semantic_img = resize(semantic_img, (self.img_height, self.img_width), order = 0, mode = 'reflect',
                                  anti_aliasing=False, preserve_range = True) # order = 0, nearest neighbour
            semantic_img = semantic_img.astype(np.uint8)
        
        # Add noise
        if self.noisy_obs is True:
            depth_img, semantic_img = self.add_noise(depth_img, semantic_img)
        
        if self.point_type == PointType.SEMANTIC:
            cloud_ros = self.cloud_generator.generate_cloud_semantic(color_img, semantic_img, depth_img,
                                                                     color_img_ros.header.stamp)
        else:
            print('Point type not supported!')

        # Publish point cloud
        self.pcl_pub.publish(cloud_ros)

    def add_noise(self, depth_img, semantic_img):
        noisy_depth_img = depth_img + np.random.normal(0, self.depth_noise_std, depth_img.shape).astype(np.float32)
        np.place(noisy_depth_img, depth_img == 0, 0)
        error_mask = np.random.sample(size=semantic_img.shape[:2]) > self.true_class_prob
        random_classes = np.random.choice(self.class_colors.shape[0],size=np.count_nonzero(error_mask))
        error_mask = np.repeat(error_mask[:,:,None],repeats=3,axis=2)
        np.place(semantic_img, error_mask, self.class_colors[random_classes, :])
        
        return noisy_depth_img, semantic_img
        

def main(args):
    rospy.init_node('semantic_cloud', anonymous=True)
    sem_cloud = SemanticCloud()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Semantic cloud shutting down!")


if __name__ == '__main__':
    main(sys.argv)
