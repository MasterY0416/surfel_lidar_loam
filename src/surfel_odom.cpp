#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"
#include "lidar_data.h"
#include "voxel_util.hpp"
#include <feature/lidar_surfel.h>
#include "sensor_data/point_data.h"

#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>



using namespace nanoflann;
using SE3d = Sophus::SE3<double>;



#define DISTORTION 0

using namespace std;

double MINIMUM_RANGE = 1.5;

const int SAMPLES_DIM = 7;

typedef std::vector<std::vector<double> > kd_tree_pointcloud;

typedef KDTreeVectorOfVectorsAdaptor< kd_tree_pointcloud, double >  my_kd_tree_t;



int corner_correspondence = 0, plane_correspondence = 0;

constexpr double SCAN_PERIOD = 0.1;
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

int skipFrameNum = 5;
int skipFrameNum1 = 5;

bool systemInited = false;

int flag = 0;

/// surfels 
std::vector<std::vector<int>> match_st_;

std::vector<point_> surfel_map_point;
std::vector<point_> surfel_cur_point;
Eigen::aligned_vector<clins::SurfelCorrespondence> surfel_correspondence_;
clins::LiDARSurfel surfel_cur_; //curr surfel scan
clins::LiDARSurfel surfel_cur_in_M; //local surfel map
clins::LiDARSurfel surfel_last_in_M; //local surfel map

pcl::KdTreeFLANN<pcl::PointXYZRGBNormal>::Ptr kdtree_surfel_map_(new pcl::KdTreeFLANN<pcl::PointXYZRGBNormal>());

pcl::KdTreeFLANN<pcl::PointXYZRGBNormal>::Ptr kdtree_surfel_map_zy(new pcl::KdTreeFLANN<pcl::PointXYZRGBNormal>());

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;

nav_msgs::Odometry laserOdometry_record;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
// pcl::PointCloud<RTPoint>::Ptr cornerPointsSharp_(new pcl::PointCloud<RTPoint>());
// RTPointCloud::Ptr cornerPointsSharp_(new RTPointCloud());
//   pcl::PointCloud<point_data::PointcloudXYZITR>::Ptr cornerPointsSharp_(
//       new pcl::PointCloud<point_data::PointcloudXYZITR>());
        pcl::PointCloud<RTPoint>::Ptr cornerPointsSharp_(
      new pcl::PointCloud<RTPoint>());

pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

// pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());

pcl::PointCloud<RTPoint>::Ptr laserCloudCornerLast(new pcl::PointCloud<RTPoint>());


pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr_1(1, 0, 0, 0);
Eigen::Vector3d t_w_curr_1(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

double parameters[7] = {0, 0, 0, 1, 0, 0, 0};

Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

Eigen::Quaterniond q_last(para_q);
Eigen::Vector3d t_last(para_t);

Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);


int map_cnt = 4;

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::mutex mBuf;

void TransformSurfel(std::vector<VoxelShape*> &in, std::vector<VoxelShape*> &out, Eigen::Matrix4d transform) {
    //Mean, transform
    //normal, just rotation
    //update feat
    //Question? when add keyframe surfel to map, how to deal with two same voxel?

    SE3d trans_se3(transform);
    for(auto const &surfel_in:in){

        Eigen::Vector3d mean_Lk = surfel_in->mean_;
        Eigen::Vector3d normal_Lk = surfel_in->normal_;
        Eigen::Vector3d mean_out;
        Eigen::Vector3d normal_out;

        mean_out = trans_se3 * mean_Lk;
        normal_out = trans_se3.rotationMatrix() * normal_Lk;

        VoxelShape *surfel_out = new VoxelShape(*surfel_in);

        surfel_out->mean_ = mean_out;
        surfel_out->normal_ = normal_out;
        out.emplace_back(surfel_out);
    }
}


void transformUpdate()
{
	q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
	t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}


void transformAssociateToMap()
{
	q_w_curr = q_wmap_wodom * q_wodom_curr;
	t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}



void removeClosedPointCloud(const RTPointCloud::Ptr &cloud_in,
                              const RTPointCloud::Ptr &cloud_out, float thres)
{
    if (&cloud_in != &cloud_out)
    {
        cloud_out->header = cloud_in->header;
        cloud_out->points.resize(cloud_in->points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in->points.size(); ++i)
    {
        if (cloud_in->points[i].x * cloud_in->points[i].x + cloud_in->points[i].y * cloud_in->points[i].y + cloud_in->points[i].z * cloud_in->points[i].z < thres * thres)
            continue;
        cloud_out->points[j] = cloud_in->points[i];
        j++;
    }
    if (j != cloud_in->points.size())
    {
        cloud_out->points.resize(j);
    }

    cloud_out->height = 1;
    cloud_out->width = static_cast<uint32_t>(j);
    cloud_out->is_dense = true;
}

void removeNaNFromPointCloud (const RTPointCloud::Ptr &cloud_in, 
                              const RTPointCloud::Ptr &cloud_out,
                              std::vector<int> &index)
{
  // If the clouds are not the same, prepare the output
  if (&cloud_in != &cloud_out)
  {
    cloud_out->header = cloud_in->header;
    cloud_out->points.resize (cloud_in->points.size ());
  }
  // Reserve enough space for the indices
  index.resize (cloud_in->points.size ());
  size_t j = 0;

  // If the data is dense, we don't need to check for NaN
  if (cloud_in->is_dense)
  {
    // Simply copy the data
    *cloud_out = *cloud_in;
    for (j = 0; j < cloud_out->points.size (); ++j)
      index[j] = static_cast<int>(j);
  }
  else
  {
    for (size_t i = 0; i < cloud_in->points.size (); ++i)
    {
      if (!pcl_isfinite (cloud_in->points[i].x) || 
          !pcl_isfinite (cloud_in->points[i].y) || 
          !pcl_isfinite (cloud_in->points[i].z))
        continue;
      cloud_out->points[j] = cloud_in->points[i];
      index[j] = static_cast<int>(i);
      j++;
    }
    if (j != cloud_in->points.size ())
    {
      // Resize to the correct size
      cloud_out->points.resize (j);
      index.resize (j);
    }

    cloud_out->height = 1;
    cloud_out->width  = static_cast<uint32_t>(j);

    // Removing bad points => dense (note: 'dense' doesn't mean 'organized')
    cloud_out->is_dense = true;
  }
}

// undistort lidar point
void TransformToStart(PointType const *const pi, PointType *const po)
{
    //interpolation ratio
    double s;
    if (DISTORTION)
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
    //s = 1;
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    Eigen::Vector3d t_point_last = s * t_last_curr;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

void TransformToStart_(RTPoint const *const pi, PointType *const po)
{
    //interpolation ratio
    double s;
    if (DISTORTION)
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
    //s = 1;
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    Eigen::Vector3d t_point_last = s * t_last_curr;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}


// transform all lidar points to the start of the next frame

void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // undistort point first
    pcl::PointXYZI un_point_tmp;
    TransformToStart(pi, &un_point_tmp);

    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->intensity = int(pi->intensity);
}


void UndistortSurfel(const std::shared_ptr<std::vector<VoxelShape *>> &surfel_raw,
                                         std::shared_ptr<std::vector<VoxelShape *>> &surfel_in_target) {
        // std::cout << "a" << std::endl;
        surfel_in_target->reserve(surfel_raw->size());
        // std::cout << "ab" << std::endl;

        // SE3d pose_G_to_target = GetLidarPose(target_timestamp).inverse();
        // std::cout << "abc" << std::endl;

        Eigen::Matrix3d rotation_matrix;
        rotation_matrix = q_last_curr.toRotationMatrix();
        Eigen::Matrix4d cur_to_target;
        cur_to_target.setIdentity();
        cur_to_target.block<3, 3>(0, 0) = rotation_matrix;
        cur_to_target.block<3, 1>(0, 3) = t_last_curr;
        cout << cur_to_target << endl;
        SE3d trans_se3(cur_to_target);
        std::size_t cnt = 0;
        for(auto const &raw_sf:*surfel_raw){
            // SE3d pose_Lk_to_G = GetLidarPose(raw_sf->time_mean_);
            // std::cout << "ab1" << std::endl;

            //TODO: ? u and mean correct?
        Eigen::Vector3d mean_Lk = raw_sf->mean_;
        Eigen::Vector3d normal_Lk = raw_sf->normal_;
        Eigen::Vector3d mean_out;
        Eigen::Vector3d normal_out;

        mean_out = trans_se3 * mean_Lk;
        normal_out = trans_se3.rotationMatrix() * normal_Lk;
        // std::cout << "ab2" << std::endl;

        VoxelShape *surfel_new = new VoxelShape(*raw_sf);
        // VoxelShape temp2 = *surfel_new;
        // std::cout << "ab3" << std::endl;

        surfel_new->mean_ = mean_out;
        surfel_new->normal_ = normal_out;
        surfel_in_target->emplace_back(surfel_new);
        }
    }



void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh; 

    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);
    nh.param<int>("skip_frame", skipFrameNum1, 2);
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);


    printf("Mapping %d Hz \n", skipFrameNum);
    //     /velodyne_points     /os_cloud_node/points
    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudSharpHandler);

    // ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);

    // ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);

    // ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);

    // ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/surfel_map", 100);

    // ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

    // ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);

     ros::Publisher pub_shape_ = nh.advertise<visualization_msgs::MarkerArray>("/shape", 10000);
     ros::Publisher pub_shape_1 = nh.advertise<visualization_msgs::MarkerArray>("/shape1", 10000);
     ros::Publisher pub_shape_2 = nh.advertise<visualization_msgs::MarkerArray>("/shape2", 10000);
     ros::Publisher pub_shape_3 = nh.advertise<visualization_msgs::MarkerArray>("/shape3", 10000);


    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);

    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    nav_msgs::Path laserPath;

    // srand(static_cast<unsigned int>(time(nullptr)));
	// kdtree_demo(1000 /* samples */, SAMPLES_DIM /* dim */);

    int frameCount = 0;
    map_cnt = skipFrameNum;
    vector<VoxelShape*> shape_vec_target;
    ros::Rate rate(100);

    while (ros::ok())
    {
        ros::spinOnce();

        if (!cornerSharpBuf.empty())
        {
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
   
            std::cout << "!!!!!" << std::endl;
            
            std::cout << cornerSharpBuf.front()->header << std::endl;
            mBuf.lock();
            cornerPointsSharp_->clear();
            cornerPointsSharp->clear();
            pcl::fromROSMsg(*(cornerSharpBuf.front()), *(cornerPointsSharp_));
            pcl::fromROSMsg(*(cornerSharpBuf.front()), *(cornerPointsSharp));
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*(cornerPointsSharp), *(cornerPointsSharp), indices);
            removeClosedPointCloud(*(cornerPointsSharp), *(cornerPointsSharp), MINIMUM_RANGE);

            cornerSharpBuf.pop();
            mBuf.unlock();

			q_wodom_curr.x() = laserOdometry_record.pose.pose.orientation.x;
			q_wodom_curr.y() = laserOdometry_record.pose.pose.orientation.y;
			q_wodom_curr.z() = laserOdometry_record.pose.pose.orientation.z;
			q_wodom_curr.w() = laserOdometry_record.pose.pose.orientation.w;
			t_wodom_curr.x() = laserOdometry_record.pose.pose.position.x;
			t_wodom_curr.y() = laserOdometry_record.pose.pose.position.y;
			t_wodom_curr.z() = laserOdometry_record.pose.pose.position.z;


            // q_wodom_curr.normalized();
            // transformAssociateToMap();

            // q_w_curr = q_wmap_wodom * q_wodom_curr;
            // t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;

            cout << q_w_curr.w() << endl;
            cout << t_w_curr << endl;

            std::cout << "///////" << std::endl;
            removeNaNFromPointCloud(cornerPointsSharp_, cornerPointsSharp_, indices);
            removeClosedPointCloud(cornerPointsSharp_, cornerPointsSharp_, MINIMUM_RANGE);             

            int cloudSize = cornerPointsSharp_->points.size();
            surfel_cur_point.clear();
            for (int i = 0;i < cloudSize; i++)
            {
                cornerPointsSharp_->points[i].intensity = cornerPointsSharp_->points[i].ring + cornerPointsSharp_->points[i].time;
                point_ p_;
                p_.point << cornerPointsSharp_->points[i].x, cornerPointsSharp_->points[i].y, cornerPointsSharp_->points[i].z;
                surfel_cur_point.emplace_back(p_);    //cur creat surfel
            }
            // surfel_map_point.clear();
            // for (int i = 0; i < cloudSize; i++)
            // {
                
            // PointType pointSel;
            // TransformToStart_(&(cornerPointsSharp_->points[i]), &pointSel);
            // point_ p;
            // p.point << pointSel.x, pointSel.y, pointSel.z;
            // surfel_map_point.emplace_back(p);      //map creat surfel
            // }
            vector<unordered_map<VOXEL_LOC, VoxelShape*>> shape_cur(map_cnt);
            int shape_map_sz_ = 0;
            buildVoxelMap(surfel_cur_point, 0.5, 10, shape_cur[0]);
            for(int map_idx = 1;map_idx < map_cnt;map_idx++){
                buildVoxelMap(shape_cur[map_idx-1], 2, 10, shape_cur[map_idx]);
                shape_map_sz_ += shape_cur[map_idx].size();
            }
            // buildVoxelMap(shape_cur[0], 2, 10, shape_cur[1]);
            // buildVoxelMap(shape_cur[1], 2, 10, shape_cur[2]);    
            // vector<VoxelShape*> shape_vec_cur;
            // shape_vec_cur.reserve(shape_map_sz_ + shape_cur[0].size());
            // for(int map_idx = 0; map_idx < map_cnt; map_idx++){
            //     for(auto iter = shape_cur[map_idx].begin(); iter!=shape_cur[map_idx].end();iter++){
            //         if(iter->second->is_plane)
            //             shape_vec_cur.emplace_back(iter->second);
            //     }
            // }
            surfel_cur_.Clear();
            int surfel_sz_ = 0;
            for(int i = 0;i < shape_cur.size();i++){
                surfel_sz_ += shape_cur.at(i).size();
            }
            surfel_cur_.surfel_vec_->reserve(surfel_sz_);
            for(int i = 0;i < shape_cur.size();i++){
                for(auto iter = shape_cur.at(i).begin(); iter != shape_cur.at(i).end(); ){
                    // if (iter->second->time_mean_ < 0) continue;
                    // if (iter->second->time_mean_ < 0){
                    //     delete iter->second;
                    //     iter->second = NULL;
                    //     iter = shape_cur.at(i).erase(iter);
                    //     continue;
                    // }
                    if (iter->second->is_plane){
                        // iter->second->time_mean_ += scan_timestamp;
                        surfel_cur_.surfel_vec_->emplace_back(iter->second);
                        iter++;
                    }
                    else{
                        delete iter->second;
                        iter->second = NULL;
                        shape_cur.at(i).erase(iter++);
                    }
                }
            }

            surfel_cur_in_M.surfel_vec_->clear();
            UndistortSurfel(surfel_cur_.surfel_vec_, surfel_cur_in_M.surfel_vec_);

            // vector<unordered_map<VOXEL_LOC, VoxelShape*>> shape_map(map_cnt);
            // int shape_map_sz = 0;
            // buildVoxelMap(surfel_map_point, 0.5, 10, shape_map[0]);
            // for(int map_idx = 1;map_idx < map_cnt;map_idx++){
            //     buildVoxelMap(shape_map[map_idx-1], 2, 10, shape_map[map_idx]);
            //     shape_map_sz += shape_map[map_idx].size();
            // }
            // // buildVoxelMap(shape_map[0], 2, 10, shape_map[1]);
            // // buildVoxelMap(shape_map[1], 2, 10, shape_map[2]);

            // surfel_cur_in_M.Clear();
            // int surfel_sz = 0;
            // for(int i = 0;i < shape_map.size();i++){
            //     surfel_sz += shape_map.at(i).size();
            // }
            // surfel_cur_in_M.surfel_vec_->reserve(surfel_sz);
            // for(int i = 0;i < shape_map.size();i++){
            //     for(auto iter = shape_map.at(i).begin(); iter != shape_map.at(i).end(); iter++){
            //         // if (iter->second->time_mean_ < 0) continue;
            //         if (iter->second->is_plane){
            //             // iter->second->time_mean_ += scan_timestamp;
            //             surfel_cur_in_M.surfel_vec_->emplace_back(iter->second);
            //         }
            //     }
            // }

            // vector<VoxelShape*> shape_vec_source;
            // shape_vec_source.reserve(shape_map_sz + shape_map[0].size());
            // for(int map_idx = 0; map_idx < map_cnt; map_idx++){
            //     for(auto iter = shape_map[map_idx].begin(); iter!=shape_map[map_idx].end();iter++){
            //         if(iter->second->is_plane)
            //             shape_vec_source.emplace_back(iter->second);
            //     }
            // }

            TicToc t_whole;
            // initializing
            if (!systemInited)
            {
                systemInited = true;
                std::cout << "Initialization finished \n";
                // shape_vec_target = shape_vec_source;
                surfel_last_in_M = surfel_cur_in_M;
                // surfel_map_point.clear();
            }
            else
            {
                int cornerPointsSharpNum = cornerPointsSharp_->points.size();
                int surfPointsFlatNum = surfPointsFlat->points.size();

        
                TicToc t_opt;
                for (size_t opti_counter = 0; opti_counter < 5; ++opti_counter)
                {
                    corner_correspondence = 0;
                    plane_correspondence = 0;

                    // ceres::LossFunction *loss_function1 = NULL;
                    // ceres::LossFunction *loss_function1 = new ceres::HuberLoss(0.1);
                    // ceres::LocalParameterization *q_parameterization1 =
                    //     new ceres::EigenQuaternionParameterization();
                    // ceres::Problem::Options problem_options1;

                    // ceres::Problem problem1(problem_options1);
                    // problem1.AddParameterBlock(para_q, 4, q_parameterization1);
                    // problem1.AddParameterBlock(para_t, 3);


					ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
					ceres::LocalParameterization *q_parameterization =
						new ceres::EigenQuaternionParameterization();
					ceres::Problem::Options problem_options;

					ceres::Problem problem(problem_options);
					problem.AddParameterBlock(parameters, 4, q_parameterization);
					problem.AddParameterBlock(parameters + 4, 3);


                    pcl::PointXYZI pointSel;
                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqDis;

                    TicToc t_data;
                    // find correspondence for corner features

                    //find correspondence
                    surfel_correspondence_.clear();

                    std::vector<std::vector<double>> feat_map_vec(surfel_last_in_M.surfel_vec_->size(), std::vector<double>(7, 0.0));
                    std::vector<std::vector<double>> feat_cur_in_m(surfel_cur_in_M.surfel_vec_->size(), std::vector<double>(7, 0.0));

                    for(int i = 0;i < surfel_last_in_M.surfel_vec_->size();i++){
                        feat_map_vec[i][0] = surfel_last_in_M.surfel_vec_->at(i)->mean_[0];
                        feat_map_vec[i][1] = surfel_last_in_M.surfel_vec_->at(i)->mean_[1];
                        feat_map_vec[i][2] = surfel_last_in_M.surfel_vec_->at(i)->mean_[2];
                        feat_map_vec[i][3] = surfel_last_in_M.surfel_vec_->at(i)->normal_[0];
                        feat_map_vec[i][4] = surfel_last_in_M.surfel_vec_->at(i)->normal_[1];
                        feat_map_vec[i][5] = surfel_last_in_M.surfel_vec_->at(i)->normal_[2];
                        feat_map_vec[i][6] = surfel_last_in_M.surfel_vec_->at(i)->voxel_size_;
                    }
                    for(int i = 0; i < surfel_cur_in_M.surfel_vec_->size();i++){
                        feat_cur_in_m[i][0] = surfel_cur_in_M.surfel_vec_->at(i)->mean_[0];
                        feat_cur_in_m[i][1] = surfel_cur_in_M.surfel_vec_->at(i)->mean_[1];
                        feat_cur_in_m[i][2] = surfel_cur_in_M.surfel_vec_->at(i)->mean_[2];
                        feat_cur_in_m[i][3] = surfel_cur_in_M.surfel_vec_->at(i)->normal_[0];
                        feat_cur_in_m[i][4] = surfel_cur_in_M.surfel_vec_->at(i)->normal_[1];
                        feat_cur_in_m[i][5] = surfel_cur_in_M.surfel_vec_->at(i)->normal_[2];
                        feat_cur_in_m[i][6] = surfel_cur_in_M.surfel_vec_->at(i)->voxel_size_;
                    }
                    KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double>>, double> kdtree_map_nano(7, feat_map_vec, 10);
                    KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double>>, double> kdtree_cur_m_nano(7, feat_cur_in_m, 10);
                    const size_t num_resultes = 1;

                    std::vector<std::vector<int>> idx_cm;
                    std::vector<std::vector<double>> dis_cm;
                    dis_cm.resize(surfel_cur_in_M.surfel_vec_->size());
                    idx_cm.resize(surfel_cur_in_M.surfel_vec_->size());
                    for(int i = 0;i < surfel_cur_.surfel_vec_->size();i++){
                        std::vector<size_t> ret_idxs_cm(num_resultes);
                        std::vector<double> out_dis_cm(num_resultes);
                        nanoflann::KNNResultSet<double> resultSet_cm(num_resultes);
                        resultSet_cm.init(&ret_idxs_cm[0], &out_dis_cm[0]);
                        std::vector<double> query_p = feat_cur_in_m[i];
                        kdtree_map_nano.index->findNeighbors(resultSet_cm, &query_p[0], nanoflann::SearchParams(10));
                        idx_cm[i].push_back(ret_idxs_cm[0]);
                        dis_cm[i].push_back(out_dis_cm[0]);
                    }
                    std::vector<std::vector<int>> pair_cm;
                    for(int c_idx = 0;c_idx < surfel_cur_.surfel_vec_->size();c_idx++){
                        int m_idx = idx_cm[c_idx][0];
                        double dix = dis_cm[c_idx][0];
                        std::vector<double> query_p = feat_map_vec[m_idx];
                        std::vector<size_t> ret_idxs_mc(num_resultes);
                        std::vector<double> out_dis_mc(num_resultes);
                        nanoflann::KNNResultSet<double> resultSet_mc(num_resultes);
                        resultSet_mc.init(&ret_idxs_mc[0], &out_dis_mc[0]);
                        kdtree_cur_m_nano.index->findNeighbors(resultSet_mc, &query_p[0], nanoflann::SearchParams(10));

                        if(ret_idxs_mc[0] == c_idx){
                            pair_cm.push_back({c_idx, m_idx});
                        }
                    }
                    match_st_.clear();
                    match_st_ = pair_cm;
                    //make surfel correspondence
                    std::cout<<"Find KDTree correspondence: "<< pair_cm.size() << " Origin KDTree correspondence : "<< feat_cur_in_m.size() << std::endl;
                    std::cout<<"Surfel cur size: " << surfel_cur_.surfel_vec_->size() << " Surfel map size: " << surfel_last_in_M.surfel_vec_->size() << std::endl;
                    for(auto &vec_cm:pair_cm){
                        if(feat_cur_in_m[vec_cm[0]][6] != feat_map_vec[vec_cm[1]][6]) continue;
                        clins::SurfelCorrespondence surfel_cor;
                        // surfel_cor.t_surfel = surfel_cur.surfel_vec_->at(vec_cm[0])->time_mean_;
                        // surfel_cor.t_map = surfel_map_.timestamp;
                        surfel_cor.cur_surfel_u = surfel_cur_.surfel_vec_->at(vec_cm[0])->mean_;
                        surfel_cor.map_surfel_u = surfel_last_in_M.surfel_vec_->at(vec_cm[1])->mean_;
            //            surfel_cor.normal_surfel = surfel_map_.surfel_vec_->at(vec_cm[1])->normal_.normalized();
            //            surfel_cor.normal_surfel = surfel_cor.normal_surfel /(sqrt(0.04 + surfel_map_.surfel_vec_->at(vec_cm[1])->eigen_val_[0]));
                        VoxelShape temp(*(surfel_last_in_M.surfel_vec_->at(vec_cm[1])));
                        // temp.update_voxel(*(surfel_cur_in_M.surfel_vec_->at(vec_cm[0])));
            ////            if (!temp.is_plane) continue;
                        surfel_cor.normal_surfel = temp.normal_.normalized();
						surfel_cor.normal_cur = surfel_cur_.surfel_vec_->at(vec_cm[0])->normal_;
                        surfel_cor.weight = 1 / sqrt(temp.eigen_val_[0] + 0.04);
    
                        // surfel_cor.weight = 1 / sqrt(temp.eigen_val_[0] + 0.04);
                        // surfel_correspondence_.push_back(surfel_cor);
                        double s;
                        s = 1.0;
						ceres::CostFunction *cost_function = LidarSurfelsFactor::Create(surfel_cor, s);
						problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);

						ceres::CostFunction *cost_function1 = LidarSurfelsRotFactor::Create(surfel_cor, s);
						problem.AddResidualBlock(cost_function1, loss_function, parameters, parameters + 4);



                    }












                    // cout << "111" << endl;
 
                    // kd_tree_pointcloud surfel_feat_pcd;
                    // int point_sz = 0;
                    // point_sz = surfel_last_in_M.surfel_vec_->size();
                    // surfel_feat_pcd.resize(point_sz);
                    // for (size_t i = 0; i < point_sz; i++)
                    // {
                    //     surfel_feat_pcd[i].resize(SAMPLES_DIM);
                    //     surfel_feat_pcd[i][0] = surfel_last_in_M.surfel_vec_->at(i)->feat_2[0];
                    //     surfel_feat_pcd[i][1] = surfel_last_in_M.surfel_vec_->at(i)->feat_2[1];
                    //     surfel_feat_pcd[i][2] = surfel_last_in_M.surfel_vec_->at(i)->feat_2[2];
                    //     surfel_feat_pcd[i][3] = surfel_last_in_M.surfel_vec_->at(i)->feat_2[3];
                    //     surfel_feat_pcd[i][4] = surfel_last_in_M.surfel_vec_->at(i)->feat_2[4];
                    //     surfel_feat_pcd[i][5] = surfel_last_in_M.surfel_vec_->at(i)->feat_2[5];
                    //     surfel_feat_pcd[i][6] = surfel_last_in_M.surfel_vec_->at(i)->feat_2[6];
                    //     // cout << surfel_feat_pcd[i][1] << " " << surfel_feat_pcd[i][2] << " " << surfel_feat_pcd[i][2] << endl;
                    //     // surfel_feat_pcd[i][0] = surfel_last_in_M.surfel_vec_->at(i)->feat_[0];
                    //     // surfel_feat_pcd[i][1] = surfel_last_in_M.surfel_vec_->at(i)->feat_[1];
                    //     // surfel_feat_pcd[i][2] = surfel_last_in_M.surfel_vec_->at(i)->feat_[2];
                    //     // surfel_feat_pcd[i][3] = surfel_last_in_M.surfel_vec_->at(i)->feat_[3];
                    //     // surfel_feat_pcd[i][4] = surfel_last_in_M.surfel_vec_->at(i)->feat_[4];
                    //     // surfel_feat_pcd[i][5] = surfel_last_in_M.surfel_vec_->at(i)->feat_[5];

                    // }
                    // cout << "s1s" << endl;
                    // // std::vector<double> (SAMPLES_DIM);
                    // // for (size_t d = 0;d < SAMPLES_DIM; d++)
                    // //     query_pt[d] = 4;                    


                    // my_kd_tree_t   kdtree_surfel_map_(SAMPLES_DIM /*dim*/, surfel_feat_pcd, 10 /* max leaf */ );

                    // cout << "s2s" << endl;


                    // kd_tree_pointcloud feat_cur_in_m;
                    // int point_sz1 = 0;
                    // point_sz1 = surfel_cur_in_M.surfel_vec_->size();
                    // feat_cur_in_m.resize(point_sz1);
                    // for (size_t i = 0; i < point_sz1; i++)
                    // {
                    //     feat_cur_in_m[i].resize(SAMPLES_DIM);
                    //     feat_cur_in_m[i][0] = surfel_cur_in_M.surfel_vec_->at(i)->feat_2[0];
                    //     feat_cur_in_m[i][1] = surfel_cur_in_M.surfel_vec_->at(i)->feat_2[1];
                    //     feat_cur_in_m[i][2] = surfel_cur_in_M.surfel_vec_->at(i)->feat_2[2];
                    //     feat_cur_in_m[i][3] = surfel_cur_in_M.surfel_vec_->at(i)->feat_2[3];
                    //     feat_cur_in_m[i][4] = surfel_cur_in_M.surfel_vec_->at(i)->feat_2[4];
                    //     feat_cur_in_m[i][5] = surfel_cur_in_M.surfel_vec_->at(i)->feat_2[5];
                    //     feat_cur_in_m[i][6] = surfel_cur_in_M.surfel_vec_->at(i)->feat_2[6];


                    //     // feat_cur_in_m[i][0] = surfel_cur_in_M.surfel_vec_->at(i)->feat_[0];
                    //     // feat_cur_in_m[i][1] = surfel_cur_in_M.surfel_vec_->at(i)->feat_[1];
                    //     // feat_cur_in_m[i][2] = surfel_cur_in_M.surfel_vec_->at(i)->feat_[2];
                    //     // feat_cur_in_m[i][3] = surfel_cur_in_M.surfel_vec_->at(i)->feat_[3];
                    //     // feat_cur_in_m[i][4] = surfel_cur_in_M.surfel_vec_->at(i)->feat_[4];
                    //     // feat_cur_in_m[i][5] = surfel_cur_in_M.surfel_vec_->at(i)->feat_[5];

                    // }


                    // // cout << "s3s" << endl;




                    // // kdtree_surfel_map_.index->findNeighbors(resultSet, &feat_cur_in_m[0], nanoflann::SearchParams(10));



                    // std::vector<std::vector<int>> idxKNN_cm(point_sz1, std::vector<int>(1));
                    // //std::vector<std::vector<int>> idxKNN_mc(feat_cur_in_m->size(), std::vector<int>(1));
                    // std::vector<std::vector<float>> disKNN_cm(point_sz1, std::vector<float>(1));
                    // // cout << "s4s" << endl;

                    // for(int i = 0;i < surfel_cur_in_M.surfel_vec_->size();i++){
                        
                    //     // pcl::PointXYZRGBNormal point_cur_in_m = feat_cur_in_m->points.at(i);
                    //     // kdtree_surfel_map_->nearestKSearch(point_cur_in_m, 1, idxKNN_cm[i], disKNN_cm[i]);
                    // // cout << "313" << endl;

	                //     std::vector<double> point_cur_in_m(SAMPLES_DIM);
                    //     for (int j = 0; j < SAMPLES_DIM; j++)
                    //         point_cur_in_m[j] = feat_cur_in_m[i][j];
                    // // cout << "121" << endl;
                    //     const size_t num_results = 1;
                    //     std::vector<size_t>  ret_indexes(num_results);
                    //     std::vector<double> out_dists_sqr(num_results);
                    //     nanoflann::KNNResultSet<double> resultSet(num_results);
                    //     resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
                    // // cout << "151" << endl;

                    // // cout << "41" << endl;

                    //     kdtree_surfel_map_.index->findNeighbors(resultSet, &point_cur_in_m[0], nanoflann::SearchParams(10));
                    // // cout << "18" << endl;
                       
                    //     idxKNN_cm[i][0] = ret_indexes[0];
                    //     disKNN_cm[i][0] = out_dists_sqr[0];
                    // }
                    // // cout << 
                    // // cout << "s5s" << endl;





                    // my_kd_tree_t kdtree_cur_in_m(SAMPLES_DIM /*dim*/, feat_cur_in_m, 10 /* max leaf */ );



                    // std::vector<std::vector<int>> pair_cm;
                    // for(int c_idx = 0; c_idx < surfel_cur_in_M.surfel_vec_->size(); c_idx++){
                    //     int m_idx = idxKNN_cm[c_idx][0];
                    //     int m_dis = disKNN_cm[c_idx][0];
                    //     std::vector<int> idxKNN_mc;
                    //     std::vector<float> disKNN_mc;
	                //     std::vector<double> point_map(SAMPLES_DIM);
                    //     point_map[0] = surfel_last_in_M.surfel_vec_->at(m_idx)->feat_2[0];
                    //     point_map[1] = surfel_last_in_M.surfel_vec_->at(m_idx)->feat_2[1];
                    //     point_map[2] = surfel_last_in_M.surfel_vec_->at(m_idx)->feat_2[2];
                    //     point_map[3] = surfel_last_in_M.surfel_vec_->at(m_idx)->feat_2[3];
                    //     point_map[4] = surfel_last_in_M.surfel_vec_->at(m_idx)->feat_2[4];
                    //     point_map[5] = surfel_last_in_M.surfel_vec_->at(m_idx)->feat_2[5];
                    //     point_map[6] = surfel_last_in_M.surfel_vec_->at(m_idx)->feat_2[6];

                    //     // point_map[0] = surfel_last_in_M.surfel_vec_->at(m_idx)->feat_[0];
                    //     // point_map[1] = surfel_last_in_M.surfel_vec_->at(m_idx)->feat_[1];
                    //     // point_map[2] = surfel_last_in_M.surfel_vec_->at(m_idx)->feat_[2];
                    //     // point_map[3] = surfel_last_in_M.surfel_vec_->at(m_idx)->feat_[3];
                    //     // point_map[4] = surfel_last_in_M.surfel_vec_->at(m_idx)->feat_[4];
                    //     // point_map[5] = surfel_last_in_M.surfel_vec_->at(m_idx)->feat_[5];

                    //     const size_t num_results1 = 1;
                    //     std::vector<size_t> ret_indexes1(num_results1);
                    //     std::vector<double> out_dists_sqr1(num_results1);
                    //     nanoflann::KNNResultSet<double> resultSet1(num_results1);
                    //     resultSet1.init(&ret_indexes1[0], &out_dists_sqr1[0]);
                    //     kdtree_cur_in_m.index->findNeighbors(resultSet1, &point_map[0], nanoflann::SearchParams(10));

                    //     if(ret_indexes1[0] == c_idx){
                    //         pair_cm.push_back({c_idx, m_idx});
                    //     }
                    // }
                    // // cout << "s6s" << endl;


                    // // cout << "222" << endl;


                    // cout <<"surfels_last nums:"<< surfel_last_in_M.surfel_vec_->size() << endl;
                    // cout <<"surfels_cur nums:"<< surfel_cur_in_M.surfel_vec_->size() << endl;

                    
                    // cout << "scanMatch" <<  pair_cm.size() << endl;
                    // cout << "sd" << endl;
 
                    // for(auto &vec_cm:pair_cm){
                    //     clins::SurfelCorrespondence surfel_cor;
                    //     // surfel_cor.t_surfel = surfel_cur.surfel_vec_->at(vec_cm[0])->time_mean_;
                    //     // surfel_cor.t_map = surfel_map_.timestamp;
                    //     // cout << "11" << endl;
                    //     surfel_cor.cur_surfel_u = surfel_cur_.surfel_vec_->at(vec_cm[0])->u_;
                    //     // cout << "22" << endl;

                    //     surfel_cor.map_surfel_u = surfel_last_in_M.surfel_vec_->at(vec_cm[1])->u_;
                    //     //TODO: need more acc normal
                    //     //TODO: use surfel_cur_in_m and surfel_map to build more acc vector
                    //     VoxelShape temp(*(surfel_last_in_M.surfel_vec_->at(vec_cm[1])));
                        

                    //     // std::cout << temp.mean_ << std::endl;
                    //     // temp.update_voxel(*(surfel_cur_in_M.surfel_vec_->at(vec_cm[0])));
                    //     surfel_cor.normal_surfel = temp.normal_.normalized();
                    //     // cout << "33" << endl;

                    //     // surfel_cor.normal_surfel = surfel_last_in_M.surfel_vec_->at(vec_cm[1])->v1_;
                    //     // surfel_cor.normal_surfel = temp.v1_;
                    //     // surfel_cor.normal_surfel /= sqrt(temp.eigen_val_[0]);
                    //     // surfel_cor.normal_surfel /= sqrt(surfel_map_.surfel_vec_->at(vec_cm[1])->eigen_val_[0]);
                    //     surfel_correspondence_.push_back(surfel_cor);
                    //     // std::cout<< "cur: " <<surfel_cor.t_surfel << " map: "<< surfel_cor.t_map <<std::endl;
                    //     // cout << "44" << endl;

                    //     double s;
                    //     // if (DISTORTION)
                    //     //     // s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                    //     // else
                    //     s = 1.0;

                    //     // ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                    //     // problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                    //     // cout << vec_cm[0] << endl;
                    //     ceres::CostFunction *cost_function = LidarSurfelsFactor::Create(surfel_cor, s);
                    //     // problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                    //     problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);

                    // }
                    // double s = 1.0;
                    // ceres::CostFunction *cost_function2 = LidarSurfelsSmothnessFactor::Create(q_last,t_last, s);
                    // problem.AddResidualBlock(cost_function2, NULL, parameters, parameters + 4);

                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 6;
                    options.minimizer_progress_to_stdout = true;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    std::cout << summary.BriefReport() << std::endl; 
                    printf("solver time %f ms \n", t_solver.toc());




                }
                printf("optimization twice time %f \n", t_opt.toc());

                // t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                // q_w_curr = q_w_curr * q_last_curr;
                // q_w_curr_1
                // t_w_curr_1 = t_w_curr_1 + q_w_curr * t_last_curr;
                // q_w_curr_1 = q_w_curr_1 * q_last_curr;
                t_w_curr_1 = t_w_curr_1 + q_w_curr_1 * t_w_curr;
                q_w_curr_1 = q_w_curr_1 * q_w_curr;


            }
            

            q_last = q_w_curr;
            t_last = t_w_curr;
            transformUpdate();

            // if (frameCount % skipFrameNum1 == 0)
            // {
            //     frameCount = 0;
            surfel_last_in_M.Clear();
            // }
            surfel_last_in_M = surfel_cur_in_M;

            // SE3d pose_cur_to_G = trajectory_->GetLidarPose(kf_time);
            // Eigen::Matrix4d cur_to_target =
            //                 (pose_G_to_target * pose_cur_to_G).matrix();
            // Eigen::Matrix3d rotation_matrix;
            // rotation_matrix = q_w_curr.toRotationMatrix();
            // Eigen::Matrix4d cur_to_target;
            // cur_to_target.setIdentity();
            // cur_to_target.block<3, 3>(0, 0) = rotation_matrix;
            // cur_to_target.block<3, 1>(0, 3) = t_w_curr;
            // cout << cur_to_target << endl;

            // std::vector<VoxelShape*> surfel_in_target;
            // surfel_in_target.reserve(surfel_cur_in_M.surfel_vec_->size());
            // TransformSurfel(*(surfel_cur_in_M.surfel_vec_),
            //                                 surfel_in_target, cur_to_target);
            // cout << "surfel in target : " << surfel_in_target.size() << endl;
            // surfel_last_in_M.surfel_vec_->insert(surfel_last_in_M.surfel_vec_->end(),
            //                                         surfel_in_target.begin(),
            //                                         surfel_in_target.end());

            


            cout << "map surfel size : " << surfel_last_in_M.surfel_vec_->size() << endl;



            // std::vector<VoxelShape*> surfel_in_target;
            // for(auto const &surfel_in : *surfel_cur_in_M.surfel_vec_)
            // {
            // Eigen::Vector3d mean_Lk = surfel_in->u_;
            // Eigen::Vector3d normal_Lk = surfel_in->v1_;
            // Eigen::Vector3d mean_out;
            // Eigen::Vector3d normal_out;

            // // mean_out = trans_se3 * mean_Lk;
            // // normal_out = trans_se3.rotationMatrix() * normal_Lk;

            // VoxelShape *surfel_out = new VoxelShape(*surfel_in);


            // surfel_out->update_feat();
            // surfel_in_target.emplace_back(surfel_out);
            // }

            //  surfel_last_in_M.surfel_vec_->insert(surfel_last_in_M.surfel_vec_->end(),surfel_in_target.begin(),surfel_in_target.end());


            TicToc t_pub;



            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/camera_init";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(timeCornerPointsSharp);


            laserOdometry.pose.pose.orientation.x = q_w_curr_1.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr_1.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr_1.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr_1.w();
            laserOdometry.pose.pose.position.x = t_w_curr_1.x();
            laserOdometry.pose.pose.position.y = t_w_curr_1.y();
            laserOdometry.pose.pose.position.z = t_w_curr_1.z();


            pubLaserOdometry.publish(laserOdometry);



            // laserOdometry_record = laserOdometry;

			laserOdometry_record.pose.pose.orientation.x = q_w_curr.x();
			laserOdometry_record.pose.pose.orientation.y = q_w_curr.y();
			laserOdometry_record.pose.pose.orientation.z = q_w_curr.z();
		    laserOdometry_record.pose.pose.orientation.w = q_w_curr.w();
			laserOdometry_record.pose.pose.position.x = t_w_curr.x();
			laserOdometry_record.pose.pose.position.y = t_w_curr.y();
			laserOdometry_record.pose.pose.position.z = t_w_curr.z();





            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "/camera_init";
            pubLaserPath.publish(laserPath);


            switch(skipFrameNum)
            {
                case 1:
                    pub_voxel(shape_cur.at(0), pub_shape_, 0);
                    break;
                case 2:
                    pub_voxel(shape_cur.at(0), pub_shape_, 0);
                    pub_voxel(shape_cur.at(1), pub_shape_1, 1);
                    break;
                case 3:
                    pub_voxel(shape_cur.at(0), pub_shape_, 0);
                    pub_voxel(shape_cur.at(1), pub_shape_1, 1);
                    pub_voxel(shape_cur.at(2), pub_shape_2, 2);
                    break;
                case 4:
                    pub_voxel(shape_cur.at(0), pub_shape_, 0);
                    pub_voxel(shape_cur.at(1), pub_shape_1, 1);
                    pub_voxel(shape_cur.at(2), pub_shape_2, 2);
                    pub_voxel(shape_cur.at(3), pub_shape_3, 3);
                    break;

            }

            if (0)
            {
                int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                }

                int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
                for (int i = 0; i < surfPointsLessFlatNum; i++)
                {
                    TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                }

                int laserCloudFullResNum = laserCloudFullRes->points.size();
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                }
            }


            if (frameCount % 5 == 0)
            {
                frameCount = 0;

                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                // pcl::toROSMsg(*cornerPointsSharp_, laserCloudCornerLast2);
                //TODO: point XYZ
                pcl::toROSMsg(*cornerPointsSharp, laserCloudCornerLast2);
                
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeCornerPointsSharp);
                laserCloudCornerLast2.header.frame_id = "/laser_odom";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

            }
            printf("publication time %f ms \n", t_pub.toc());
            printf("whole laserOdometry time %f ms \n \n", t_whole.toc());
            if(t_whole.toc() > 100)
                ROS_WARN("odometry process over 100ms");
            frameCount++;
        }


        rate.sleep();
    }
    return 0;
}