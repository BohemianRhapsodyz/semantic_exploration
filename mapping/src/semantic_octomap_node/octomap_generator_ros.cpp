#include <semantic_octomap_node/octomap_generator_ros.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/impl/transforms.hpp>
#include <octomap_msgs/conversions.h>
#include <nav_msgs/OccupancyGrid.h>
#include <pcl/conversions.h>
#include <cmath>
#include <sstream>
#include <cstring> // For std::memcpy

OctomapGeneratorNode::OctomapGeneratorNode(ros::NodeHandle& nh): nh_(nh)
{
    // Initiate octree
    ROS_INFO("Semantic octomap generated!");
    octomap_generator_ = new OctomapGenerator<PCLSemantics, SemanticOctree>();
    toggle_color_service_ = nh_.advertiseService("toggle_use_semantic_color", &OctomapGeneratorNode::toggleUseSemanticColor, this);
    RLE_service_ = nh_.advertiseService("querry_RLE", &OctomapGeneratorNode::querry_RLE, this);

    reset();
    fullmap_pub_ = nh_.advertise<octomap_msgs::Octomap>("octomap_full", 1, true);
    occ_map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("occupancy_map_2D", 1, true);
    odom_tf_sub_ = nh_.subscribe("/odometry", 100, &OctomapGeneratorNode::broadcasterCallback, this);
    pointcloud_sub_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh_, pointcloud_topic_, 100));
    odom_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(nh_, "/odometry", 100));
    sync_image_odom_.reset(new message_filters::Synchronizer<SyncPolicyImageOdom>(
            SyncPolicyImageOdom(100), *pointcloud_sub_, *odom_sub_));
    sync_image_odom_->registerCallback(boost::bind(&OctomapGeneratorNode::insertCloudCallback, this, _1, _2));

}

OctomapGeneratorNode::~OctomapGeneratorNode() {}
/// Clear octomap and reset values to paramters from parameter server
void OctomapGeneratorNode::reset()
{
    nh_.getParam("/octomap/pointcloud_topic", pointcloud_topic_);
    nh_.getParam("/octomap/world_frame_id", world_frame_id_);
    nh_.getParam("/octomap/resolution", resolution_);
    nh_.getParam("/octomap/max_range", max_range_);
    nh_.getParam("/octomap/raycast_range", raycast_range_);
    nh_.getParam("/octomap/clamping_thres_min", clamping_thres_min_);
    nh_.getParam("/octomap/clamping_thres_max", clamping_thres_max_);
    nh_.getParam("/octomap/occupancy_thres", occupancy_thres_);
    nh_.getParam("/octomap/prob_hit", prob_hit_);
    nh_.getParam("/octomap/prob_miss", prob_miss_);
    nh_.getParam("/octomap/psi", psi_);
    nh_.getParam("/octomap/phi", phi_);
    nh_.getParam("/octomap/publish_2d_map", publish_2d_map);
    nh_.getParam("/octomap/min_ground_z", min_ground_z);
    nh_.getParam("/octomap/max_ground_z", max_ground_z);
    octomap_generator_->setClampingThresMin(clamping_thres_min_);
    octomap_generator_->setClampingThresMax(clamping_thres_max_);
    octomap_generator_->setResolution(resolution_);
    octomap_generator_->setOccupancyThres(occupancy_thres_);
    octomap_generator_->setProbHit(prob_hit_);
    octomap_generator_->setProbMiss(prob_miss_);
    octomap_generator_->setPsi(psi_);
    octomap_generator_->setPhi(phi_);
    octomap_generator_->setRayCastRange(raycast_range_);
    octomap_generator_->setMaxRange(max_range_);
}

bool OctomapGeneratorNode::toggleUseSemanticColor(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
    octomap_generator_->setUseSemanticColor(!octomap_generator_->isUseSemanticColor());
    if(octomap_generator_->isUseSemanticColor())
        ROS_INFO("Using semantic color");
    else
        ROS_INFO("Using rgb color");
    if (octomap_msgs::fullMapToMsg(*octomap_generator_->getOctree(), map_msg_))
        fullmap_pub_.publish(map_msg_);
    else
        ROS_ERROR("Error serializing OctoMap");
    return true;
}

bool OctomapGeneratorNode::querry_RLE(semantic_octomap::GetRLE::Request& request, semantic_octomap::GetRLE::Response& response)
{
    const octomap::point3d origin(request.origin.x, request.origin.y, request.origin.z);

    for (int i = 0; i < (int)request.endPoints.size(); ++i)
    {   
        const octomap::point3d endPoint(request.endPoints[i].x, request.endPoints[i].y, request.endPoints[i].z);
        semantic_octomap::RayRLE rayRLE_msg;
        if (octomap_generator_->get_ray_RLE(origin, endPoint, rayRLE_msg))
        {          
            response.RLE_list.push_back(rayRLE_msg);
        } 
    }
    
    return true;
}

void OctomapGeneratorNode::broadcasterCallback(const nav_msgs::Odometry::ConstPtr& odom)
{
    static tf::TransformBroadcaster br;
    tf::Transform tf;
    tf::Quaternion q_;

    q_[0] = odom->pose.pose.orientation.x;
    q_[1] = odom->pose.pose.orientation.y;
    q_[2] = odom->pose.pose.orientation.z;
    q_[3] = odom->pose.pose.orientation.w;

    nav_msgs::Odometry odom_robot;
    odom_robot.pose.pose.orientation.x = q_[0];
    odom_robot.pose.pose.orientation.y = q_[1];
    odom_robot.pose.pose.orientation.z = q_[2];
    odom_robot.pose.pose.orientation.w = q_[3];
    odom_robot.pose.pose.position.x = odom->pose.pose.position.x;
    odom_robot.pose.pose.position.y = odom->pose.pose.position.y;
    odom_robot.pose.pose.position.z = odom->pose.pose.position.z;
    geometry_msgs::Pose odom_pose = odom_robot.pose.pose;
    tf::poseMsgToTF(odom_pose, tf);
    tf::StampedTransform RobotToWorldTf(tf,odom->header.stamp,"/map","/husky/base");
    br.sendTransform(RobotToWorldTf);
}

void OctomapGeneratorNode::insertCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg, const nav_msgs::Odometry::ConstPtr& odom)
{
    pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2 ());
    pcl_conversions::toPCL(*cloud_msg, *cloud);

    tf::Transform tf;
    tf::Quaternion q_;
    tf::Quaternion q = tf::createQuaternionFromRPY(-M_PI_2,0,-M_PI_2);

    q_[0] = odom->pose.pose.orientation.x;
    q_[1] = odom->pose.pose.orientation.y;
    q_[2] = odom->pose.pose.orientation.z;
    q_[3] = odom->pose.pose.orientation.w;
    q_ = q_*q;
    nav_msgs::Odometry odom_cam;
    odom_cam.pose.pose.orientation.x = q_[0];
    odom_cam.pose.pose.orientation.y = q_[1];
    odom_cam.pose.pose.orientation.z = q_[2];
    odom_cam.pose.pose.orientation.w = q_[3];
    odom_cam.pose.pose.position.x = odom->pose.pose.position.x;
    odom_cam.pose.pose.position.y = odom->pose.pose.position.y;
    odom_cam.pose.pose.position.z = odom->pose.pose.position.z;
    geometry_msgs::Pose odom_pose = odom_cam.pose.pose;
    tf::poseMsgToTF(odom_pose, tf);
    tf::StampedTransform sensorToWorldTf(tf,odom->header.stamp,"/world","/camera");
    Eigen::Matrix4f sensorToWorld;

    pcl_ros::transformAsMatrix(sensorToWorldTf, sensorToWorld);
    octomap_generator_->insertPointCloud(cloud, sensorToWorld);
    
    // Publish full octomap
    map_msg_.header.frame_id = world_frame_id_;
    map_msg_.header.stamp = cloud_msg->header.stamp;
    if (octomap_msgs::fullMapToMsg(*octomap_generator_->getOctree(), map_msg_))
        fullmap_pub_.publish(map_msg_);
    else
        ROS_ERROR("Error serializing full OctoMap");
    
    // Publish 2D occupancy map
    if (publish_2d_map)
        publish2DOccupancyMap(octomap_generator_->getOctree(), cloud_msg->header.stamp, world_frame_id_);
}

void OctomapGeneratorNode::publish2DOccupancyMap(const SemanticOctree* octomap,
                                                 const ros::Time& stamp,
                                                 const std::string& frame_id)
{
  // get dimensions of octree
  double minX, minY, minZ, maxX, maxY, maxZ;
  octomap->getMetricMin(minX, minY, minZ);
  octomap->getMetricMax(maxX, maxY, maxZ);
  octomap::point3d minPt = octomap::point3d(minX, minY, minZ);

  unsigned int tree_depth = octomap->getTreeDepth();

  octomap::OcTreeKey paddedMinKey = octomap->coordToKey(minPt);

  nav_msgs::OccupancyGrid::Ptr occupancy_map (new nav_msgs::OccupancyGrid());

  unsigned int width, height;
  double res;

  unsigned int ds_shift = tree_depth-16;

  occupancy_map->header.stamp = stamp;
  occupancy_map->header.frame_id = frame_id;
  occupancy_map->info.resolution = res = octomap->getNodeSize(16);
  occupancy_map->info.width = width = (maxX-minX) / res + 1;
  occupancy_map->info.height = height = (maxY-minY) / res + 1;
  occupancy_map->info.origin.position.x = minX  - (res / (float)(1<<ds_shift) ) + res;
  occupancy_map->info.origin.position.y = minY  - (res / (float)(1<<ds_shift) );

  occupancy_map->data.clear();
  occupancy_map->data.resize(width*height, -1);

    // traverse all leafs in the tree:
  unsigned int treeDepth = std::min<unsigned int>(16, octomap->getTreeDepth());
  for (typename SemanticOctree::iterator it = octomap->begin(treeDepth), end = octomap->end(); it != end; ++it)
  {
  
    double node_z = it.getZ();
    double node_half_side = pow(it.getSize(), 1/3) / 2;
    double top_side = node_z + node_half_side;
    double bottom_side = node_z - node_half_side;
    
    if((bottom_side >= min_ground_z && bottom_side <= max_ground_z) ||
       (top_side >= min_ground_z && top_side <= max_ground_z)
       ||(bottom_side <= min_ground_z && top_side >= max_ground_z)
       )
    {
      bool occupied = octomap->isNodeOccupied(*it);
      int intSize = 1 << (16 - it.getDepth());

      octomap::OcTreeKey minKey=it.getIndexKey();

      for (int dx = 0; dx < intSize; dx++)
      {
        for (int dy = 0; dy < intSize; dy++)
        {
          int posX = std::max<int>(0, minKey[0] + dx - paddedMinKey[0]);
          posX>>=ds_shift;

          int posY = std::max<int>(0, minKey[1] + dy - paddedMinKey[1]);
          posY>>=ds_shift;

          int idx = width * posY + posX;

          if (occupied)
            occupancy_map->data[idx] = 100;
          else if (occupancy_map->data[idx] == -1)
          {
            occupancy_map->data[idx] = 0;
          }

        }
      }
    }
  }

  occ_map_pub_.publish(*occupancy_map);
}

bool OctomapGeneratorNode::save(const char* filename) const
{
    return octomap_generator_->save(filename);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "octomap_generator");
    ros::NodeHandle nh;
    OctomapGeneratorNode octomapGeneratorNode(nh);
    ros::spin();
    std::string save_path;
    nh.getParam("/octomap/save_path", save_path);

    octomapGeneratorNode.save(save_path.c_str());
    ROS_INFO("OctoMap saved.");
    return 0;
}
