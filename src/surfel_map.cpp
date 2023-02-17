#include <math.h>
#include <vector>
#include <aloam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>

#include "lidarFactor.hpp"
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"


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

using namespace std;

int frameCount = 0;

double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;


int laserCloudCenWidth = 10;
int laserCloudCenHeight = 10;
int laserCloudCenDepth = 5;
const int laserCloudWidth = 21;
const int laserCloudHeight = 21;
const int laserCloudDepth = 11;


const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; //4851


int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

// input: from odom
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());

// pcl::PointCloud<RTPoint>::Ptr laserCloudCornerLast(new pcl::PointCloud<RTPoint>());

pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

// ouput: all visualble cube points
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());

//input & output: points in one frame. local --> global
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
// pcl::PointCloud<RTPoint>::Ptr laserCloudFullRes(new pcl::PointCloud<RTPoint>());


// points in every cube
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());


//voxel_map 
std::vector<std::vector<int>> match_st_;
typedef std::vector<std::vector<double> > VoxelFeature;

typedef KDTreeVectorOfVectorsAdaptor< VoxelFeature, double > KdtreeNanoFLann;
int map_cnt = 4;
std::vector<point_> surfel_local_map_point;
std::vector<point_> surfel_cur_scan_point;
std::vector<point_> surfel_cur_in_map;
std::vector<point_> surfel_cur_point;
using SE3d = Sophus::SE3<double>;



clins::LiDARSurfel surfel_local_map_; 
clins::LiDARSurfel surfel_cur_in_local_map; 
clins::LiDARSurfel surfel_cur_; 


//Ceres 

double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);

double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);



std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterSurf;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;

nav_msgs::Path laserAfterMappedPath;

// set initial guess
void transformAssociateToMap()
{
	q_w_curr = q_wmap_wodom * q_wodom_curr;
	t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

void transformUpdate()
{
	q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
	t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
}

void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
	po->x = point_curr.x();
	po->y = point_curr.y();
	po->z = point_curr.z();
	po->intensity = pi->intensity;
}

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

void UndistortSurfel(const std::shared_ptr<std::vector<VoxelShape *>> &surfel_raw,
                                         std::shared_ptr<std::vector<VoxelShape *>> &surfel_in_target) {
        // std::cout << "a" << std::endl;
        surfel_in_target->reserve(surfel_raw->size());
        // std::cout << "ab" << std::endl;

        // SE3d pose_G_to_target = GetLidarPose(target_timestamp).inverse();
        // std::cout << "abc" << std::endl;
		q_w_curr.normalize();
        Eigen::Matrix3d rotation_matrix;
        rotation_matrix = q_w_curr.toRotationMatrix();
        Eigen::Matrix4d cur_to_target;
        cur_to_target.setIdentity();
        cur_to_target.block<3, 3>(0, 0) = rotation_matrix;
        cur_to_target.block<3, 1>(0, 3) = t_w_curr;
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



void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
{
	mBuf.lock();
	cornerLastBuf.push(laserCloudCornerLast2);
	mBuf.unlock();
}


void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
	mBuf.lock();
	fullResBuf.push(laserCloudFullRes2);
	mBuf.unlock();
}

//receive odomtry
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(laserOdometry);
	mBuf.unlock();

	// high frequence publish
	Eigen::Quaterniond q_wodom_curr;
	Eigen::Vector3d t_wodom_curr;
	q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
	q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
	q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
	q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
	t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
	t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
	t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

	Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
	Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; 

	nav_msgs::Odometry odomAftMapped;
	odomAftMapped.header.frame_id = "/camera_init";
	odomAftMapped.child_frame_id = "/aft_mapped";
	odomAftMapped.header.stamp = laserOdometry->header.stamp;
	odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
	odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
	odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
	odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
	odomAftMapped.pose.pose.position.x = t_w_curr.x();
	odomAftMapped.pose.pose.position.y = t_w_curr.y();
	odomAftMapped.pose.pose.position.z = t_w_curr.z();
	pubOdomAftMappedHighFrec.publish(odomAftMapped);
}

void process()
{
	while(1)
	{
		while (!cornerLastBuf.empty() &&
			!fullResBuf.empty() && !odometryBuf.empty())
		{
			mBuf.lock();
			while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				odometryBuf.pop();
			if (odometryBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				fullResBuf.pop();
			if (fullResBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
			// timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
			timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();

			timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();

			if (timeLaserCloudCornerLast != timeLaserOdometry ||
				timeLaserCloudFullRes != timeLaserOdometry)
			{
				printf("time corner %f surf %f full %f odom %f \n", timeLaserCloudCornerLast, timeLaserCloudSurfLast, timeLaserCloudFullRes, timeLaserOdometry);
				printf("unsync messeage!");
				mBuf.unlock();
				break;
			}

			laserCloudCornerLast->clear();
			pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
			cornerLastBuf.pop();



			laserCloudFullRes->clear();
			pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
			fullResBuf.pop();

			q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
			q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
			q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
			q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
			t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
			t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
			t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
			odometryBuf.pop();

			while(!cornerLastBuf.empty())
			{
				cornerLastBuf.pop();
				printf("drop lidar frame in mapping for real time performance \n");
			}

			mBuf.unlock();

			TicToc t_whole;

			transformAssociateToMap();

			TicToc t_shift;
			int centerCubeI = int((t_w_curr.x() + 5.0) / 10.0) + laserCloudCenWidth;
			int centerCubeJ = int((t_w_curr.y() + 5.0) / 10.0) + laserCloudCenHeight;
			int centerCubeK = int((t_w_curr.z() + 5.0) / 10.0) + laserCloudCenDepth;

			if (t_w_curr.x() + 5.0 < 0)
				centerCubeI--;
			if (t_w_curr.y() + 5.0 < 0)
				centerCubeJ--;
			if (t_w_curr.z() + 5.0 < 0)
				centerCubeK--;
            // find local map   
			while (centerCubeI < 3)
			{
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{ 
						int i = laserCloudWidth - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]; 
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; i >= 1; i--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeI++;
				laserCloudCenWidth++;
			}

			while (centerCubeI >= laserCloudWidth - 3)
			{ 
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int i = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; i < laserCloudWidth - 1; i++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeI--;
				laserCloudCenWidth--;
			}

			while (centerCubeJ < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = laserCloudHeight - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j >= 1; j--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ++;
				laserCloudCenHeight++;
			}

			while (centerCubeJ >= laserCloudHeight - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j < laserCloudHeight - 1; j++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ--;
				laserCloudCenHeight--;
			}

			while (centerCubeK < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = laserCloudDepth - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k >= 1; k--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK++;
				laserCloudCenDepth++;
			}

			while (centerCubeK >= laserCloudDepth - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k < laserCloudDepth - 1; k++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK--;
				laserCloudCenDepth--;
			}

			int laserCloudValidNum = 0;
			int laserCloudSurroundNum = 0;

			for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
			{
				for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
				{
					for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
					{
						if (i >= 0 && i < laserCloudWidth &&
							j >= 0 && j < laserCloudHeight &&
							k >= 0 && k < laserCloudDepth)
						{ 
							laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudValidNum++;
							laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudSurroundNum++;
						}
					}
				}
			}

			laserCloudCornerFromMap->clear();
			laserCloudSurfFromMap->clear();
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				*laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];
				*laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
			}
			int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
			int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();

            //finded local map to laserCloudCornerFromMap

            //TODO: laserCloudCornerFromMap   
            surfel_local_map_point.clear();
            for (int i = 0;i < laserCloudCornerFromMapNum; i++)
            {
                point_ p_;
                p_.point << laserCloudCornerFromMap->points[i].x, laserCloudCornerFromMap->points[i].y, laserCloudCornerFromMap->points[i].z;
                surfel_local_map_point.emplace_back(p_);    //cur creat surfel
            }

            vector<unordered_map<VOXEL_LOC, VoxelShape*>> shape_local_map(map_cnt);
            int shape_map_sz_ = 0;
            buildVoxelMap(surfel_local_map_point, 0.5, 10, shape_local_map[0]);
            for(int map_idx = 1;map_idx < map_cnt;map_idx++){
                buildVoxelMap(shape_local_map[map_idx-1], 2, 10, shape_local_map[map_idx]);
                shape_map_sz_ += shape_local_map[map_idx].size();
            }

            surfel_local_map_.Clear();
            int surfel_sz = 0;
            for(int i = 0;i < shape_local_map.size();i++){
                surfel_sz += shape_local_map.at(i).size();
            }
            surfel_local_map_.surfel_vec_->reserve(surfel_sz);
            for(int i = 0;i < shape_local_map.size();i++){
                for(auto iter = shape_local_map.at(i).begin(); iter != shape_local_map.at(i).end(); iter++){
                    // if (iter->second->time_mean_ < 0) continue;
                    if (iter->second->is_plane){
                        // iter->second->time_mean_ += scan_timestamp;
                        surfel_local_map_.surfel_vec_->emplace_back(iter->second);
                    }
                }
            }   

            //
			pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
			downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
			downSizeFilterCorner.filter(*laserCloudCornerStack);
			int laserCloudCornerStackNum = laserCloudCornerStack->points.size();
			
			int laserCloudCornerLastNum = laserCloudCornerLast->points.size();
			//cur scan downsmple

            //change cur scan voxelmap
            // function change
			
            // surfel_cur_in_map.clear();
            // for (int i = 0;i < laserCloudCornerStackNum; i++)
            // {
            //     pointOri = laserCloudCornerStack->points[i];
			// 	//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
			// 	pointAssociateToMap(&pointOri, &pointSel);
            //     point_ p_;
            //     p_.point << pointSel.x, pointSel.y, pointSel.z;
            //     surfel_cur_in_map.emplace_back(p_);    //cur creat surfel
            // }

            // vector<unordered_map<VOXEL_LOC, VoxelShape*>> shape_cur_in_map(map_cnt);
            // int shape_map_sz = 0;
            // buildVoxelMap(surfel_cur_in_map, 0.5, 10, shape_cur_in_map[0]);
            // for(int map_idx = 1;map_idx < map_cnt;map_idx++){
            //     buildVoxelMap(shape_cur_in_map[map_idx-1], 2, 10, shape_cur_in_map[map_idx]);
            //     shape_map_sz += shape_cur_in_map[map_idx].size();
            // }

            // surfel_cur_in_local_map.Clear();
            // int surfel = 0;
            // for(int i = 0;i < shape_cur_in_map.size();i++){
            //     surfel += shape_cur_in_map.at(i).size();
            // }
            // surfel_cur_in_local_map.surfel_vec_->reserve(surfel);
            // for(int i = 0;i < shape_cur_in_map.size();i++){
            //     for(auto iter = shape_cur_in_map.at(i).begin(); iter != shape_cur_in_map.at(i).end(); iter++){
            //         // if (iter->second->time_mean_ < 0) continue;
            //         if (iter->second->is_plane){
            //             // iter->second->time_mean_ += scan_timestamp;
            //             surfel_cur_in_local_map.surfel_vec_->emplace_back(iter->second);
            //         }
            //     }
            // }   


            surfel_cur_point.clear();
			for (int i = 0;i < laserCloudCornerLastNum; i++)
            {
                // pointOri = laserCloudCornerStack->points[i];
				// //double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
				// pointAssociateToMap(&pointOri, &pointSel);
                point_ p_;
                p_.point << laserCloudCornerLast->points[i].x, laserCloudCornerLast->points[i].y, 
                                                                laserCloudCornerLast->points[i].z;
                surfel_cur_point.emplace_back(p_);    //cur creat surfel
            }
			//降采样
            // for (int i = 0;i < laserCloudCornerStackNum; i++)
            // {
            //     // pointOri = laserCloudCornerStack->points[i];
			// 	// //double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
			// 	// pointAssociateToMap(&pointOri, &pointSel);
            //     point_ p_;
            //     p_.point << laserCloudCornerStack->points[i].x, laserCloudCornerStack->points[i].y, 
            //                                                     laserCloudCornerStack->points[i].z;
            //     surfel_cur_point.emplace_back(p_);    //cur creat surfel
            // }

            vector<unordered_map<VOXEL_LOC, VoxelShape*>> shape_cur(map_cnt);
            int shape_map_ = 0;
            buildVoxelMap(surfel_cur_point, 0.5, 10, shape_cur[0]);
            for(int map_idx = 1;map_idx < map_cnt;map_idx++){
                buildVoxelMap(shape_cur[map_idx-1], 2, 10, shape_cur[map_idx]);
                shape_map_ += shape_cur[map_idx].size();
            }

            surfel_cur_.Clear();
            int surfel_sz_ = 0;
            for(int i = 0;i < shape_cur.size();i++){
                surfel_sz_ += shape_cur.at(i).size();
            }
            surfel_cur_.surfel_vec_->reserve(surfel_sz_);
            for(int i = 0;i < shape_cur.size();i++){
                for(auto iter = shape_cur.at(i).begin(); iter != shape_cur.at(i).end(); iter++){
                    // if (iter->second->time_mean_ < 0) continue;
				// if (iter->second->time_mean_ < 0){
                //         delete iter->second;
                //         iter->second = NULL;
                //         iter = shape_cur.at(i).erase(iter);
                //     continue;
                // }
                    if (iter->second->is_plane){
                        // iter->second->time_mean_ += scan_timestamp;
                        surfel_cur_.surfel_vec_->emplace_back(iter->second);
                    }
					else{
                        delete iter->second;
                        // iter->second = NULL;
                        // iter = shape_cur.at(i).erase(iter);
					}
                }
            }   

            surfel_cur_in_local_map.surfel_vec_->clear();
            UndistortSurfel(surfel_cur_.surfel_vec_, surfel_cur_in_local_map.surfel_vec_);



			pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
			downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
			downSizeFilterSurf.filter(*laserCloudSurfStack);
			int laserCloudSurfStackNum = laserCloudSurfStack->points.size();

			printf("map prepare time %f ms\n", t_shift.toc());
			printf("map corner num %d  surf num %d \n", laserCloudCornerFromMapNum, laserCloudSurfFromMapNum);
			if (laserCloudCornerFromMapNum > 10 )
			{
				TicToc t_opt;
				TicToc t_tree;
                cout << "1" << endl;
                VoxelFeature feat_map_vec(surfel_local_map_.surfel_vec_->size(), std::vector<double>(7, 0.0));
                VoxelFeature feat_cur_in_m(surfel_cur_in_local_map.surfel_vec_->size(), std::vector<double>(7, 0.0));
                cout << "2" << endl;

                for(int i = 0;i < surfel_local_map_.surfel_vec_->size();i++){
                    feat_map_vec[i][0] = surfel_local_map_.surfel_vec_->at(i)->mean_[0];
                    feat_map_vec[i][1] = surfel_local_map_.surfel_vec_->at(i)->mean_[1];
                    feat_map_vec[i][2] = surfel_local_map_.surfel_vec_->at(i)->mean_[2];
                    feat_map_vec[i][3] = surfel_local_map_.surfel_vec_->at(i)->normal_[0];
                    feat_map_vec[i][4] = surfel_local_map_.surfel_vec_->at(i)->normal_[1];
                    feat_map_vec[i][5] = surfel_local_map_.surfel_vec_->at(i)->normal_[2];
                    feat_map_vec[i][6] = surfel_local_map_.surfel_vec_->at(i)->voxel_size_;
                }
                cout << "3" << endl;

                for(int i = 0; i < surfel_cur_in_local_map.surfel_vec_->size();i++){
                    feat_cur_in_m[i][0] = surfel_cur_in_local_map.surfel_vec_->at(i)->mean_[0];
                    feat_cur_in_m[i][1] = surfel_cur_in_local_map.surfel_vec_->at(i)->mean_[1];
                    feat_cur_in_m[i][2] = surfel_cur_in_local_map.surfel_vec_->at(i)->mean_[2];
                    feat_cur_in_m[i][3] = surfel_cur_in_local_map.surfel_vec_->at(i)->normal_[0];
                    feat_cur_in_m[i][4] = surfel_cur_in_local_map.surfel_vec_->at(i)->normal_[1];
                    feat_cur_in_m[i][5] = surfel_cur_in_local_map.surfel_vec_->at(i)->normal_[2];
                    feat_cur_in_m[i][6] = surfel_cur_in_local_map.surfel_vec_->at(i)->voxel_size_;
                }
                cout << "4" << endl;
                KdtreeNanoFLann kdtree_map_nano(7, feat_map_vec, 10);
                KdtreeNanoFLann kdtree_cur_m_nano(7, feat_cur_in_m, 10);
                cout << "5" << endl;


				// kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
				// kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
				printf("build tree time %f ms \n", t_tree.toc());

				for (int iterCount = 0; iterCount < 2; iterCount++)
				{
					//ceres::LossFunction *loss_function = NULL;
					ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
					ceres::LocalParameterization *q_parameterization =
						new ceres::EigenQuaternionParameterization();
					ceres::Problem::Options problem_options;

					ceres::Problem problem(problem_options);
					problem.AddParameterBlock(parameters, 4, q_parameterization);
					problem.AddParameterBlock(parameters + 4, 3);

					TicToc t_data;
					int corner_num = 0;

                    //TODO: cur to map voxmap corre


                cout << "6" << endl;

                    const size_t num_resultes = 1;
                    std::vector<std::vector<int>> idx_cm;
                    std::vector<std::vector<double>> dis_cm;
                    dis_cm.resize(surfel_cur_in_local_map.surfel_vec_->size());
                    idx_cm.resize(surfel_cur_in_local_map.surfel_vec_->size());
                    for(int i = 0;i < surfel_cur_in_local_map.surfel_vec_->size();i++){
                        std::vector<size_t> ret_idxs_cm(num_resultes);
                        std::vector<double> out_dis_cm(num_resultes);
                        nanoflann::KNNResultSet<double> resultSet_cm(num_resultes);
                        resultSet_cm.init(&ret_idxs_cm[0], &out_dis_cm[0]);
                        std::vector<double> query_p = feat_cur_in_m[i];
                        kdtree_map_nano.index->findNeighbors(resultSet_cm, &query_p[0], nanoflann::SearchParams(10));
                        idx_cm[i].push_back(ret_idxs_cm[0]);
                        dis_cm[i].push_back(out_dis_cm[0]);
                    }
                    cout << "7" << endl;

                    std::vector<std::vector<int>> pair_cm;
                    for(int c_idx = 0;c_idx < surfel_cur_in_local_map.surfel_vec_->size();c_idx++){
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
                    cout << "8" << endl;

                    //make surfel correspondence
                    std::cout<<"Find KDTree correspondence: "<< pair_cm.size() << " Origin KDTree correspondence : "<< feat_cur_in_m.size() << std::endl;
                    std::cout<<"Surfel cur size: " << surfel_cur_.surfel_vec_->size() << " Surfel map size: " << surfel_local_map_.surfel_vec_->size() << std::endl;
                    for(auto &vec_cm:pair_cm){
                        // if(feat_cur_in_m[vec_cm[0]][6] != feat_map_vec[vec_cm[1]][6]) continue;
                        clins::SurfelCorrespondence surfel_cor;
                        // surfel_cor.t_surfel = surfel_cur_.surfel_vec_->at(vec_cm[0])->time_mean_;
                        // surfel_cor.t_map = surfel_map_.timestamp;
                        surfel_cor.cur_surfel_u = surfel_cur_.surfel_vec_->at(vec_cm[0])->mean_;
                        surfel_cor.map_surfel_u = surfel_local_map_.surfel_vec_->at(vec_cm[1])->mean_;
                        //            surfel_cor.normal_surfel = surfel_map_.surfel_vec_->at(vec_cm[1])->normal_.normalized();
                        //            surfel_cor.normal_surfel = surfel_cor.normal_surfel /(sqrt(0.04 + surfel_map_.surfel_vec_->at(vec_cm[1])->eigen_val_[0]));
                        VoxelShape temp(*(surfel_local_map_.surfel_vec_->at(vec_cm[1])));
                        // temp.update_voxel(*(surfel_cur_in_lolaserCloudFullRes3cal_map.surfel_vec_->at(vec_cm[0])));
                        ////            if (!temp.is_plane) continue;
                        surfel_cor.normal_surfel = temp.normal_.normalized();
						surfel_cor.normal_cur = surfel_cur_.surfel_vec_->at(vec_cm[0])->normal_;

                        surfel_cor.weight = 1 / sqrt(temp.eigen_val_[0] + 0.04);
                        // surfel_correspondence_.push_back(surfel_cor);
                        double s;
                        s = 1.0;
						ceres::CostFunction *cost_function = LidarSurfelsFactor::Create(surfel_cor, s);
						problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);

						ceres::CostFunction *cost_function1 = LidarSurfelsRotFactor::Create(surfel_cor, s);
						problem.AddResidualBlock(cost_function1, loss_function, parameters, parameters + 4);


                    }
                    // std::cout<<"Find Surfel correspondence: "<< surfel_correspondence_.size() << " Find KDTree correspondence: "<< pair_cm.size() << std::endl;
                    cout << "9" << endl;

					//printf("corner num %d used corner num %d \n", laserCloudCornerStackNum, corner_num);
					//printf("surf num %d used surf num %d \n", laserCloudSurfStackNum, surf_num);

					printf("mapping data assosiation time %f ms \n", t_data.toc());

					TicToc t_solver;
					ceres::Solver::Options options;
					options.linear_solver_type = ceres::DENSE_QR;
					options.max_num_iterations = 4;
					options.minimizer_progress_to_stdout = false;
					options.check_gradients = false;
					options.gradient_check_relative_precision = 1e-4;
					ceres::Solver::Summary summary;
					ceres::Solve(options, &problem, &summary);
					printf("mapping solver time %f ms \n", t_solver.toc());

					//printf("time %f \n", timeLaserOdometry);
					//printf("corner factor num %d surf factor num %d\n", corner_num, surf_num);
					//printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
					//	   parameters[4], parameters[5], parameters[6]);
				}
				printf("mapping optimization time %f \n", t_opt.toc());
			}
			else
			{
				ROS_WARN("time Map corner and surf num are not enough");
			}
			transformUpdate();

			TicToc t_add;

            //TODO: get local map   降采样
			// for (int i = 0; i < laserCloudCornerStackNum; i++)
			// {
			// 	pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);

			// 	int cubeI = int((pointSel.x + 5.0) / 10.0) + laserCloudCenWidth;
			// 	int cubeJ = int((pointSel.y + 5.0) / 10.0) + laserCloudCenHeight;
			// 	int cubeK = int((pointSel.z + 5.0) / 10.0) + laserCloudCenDepth;

			// 	if (pointSel.x + 5.0 < 0)
			// 		cubeI--;
			// 	if (pointSel.y + 5.0 < 0)
			// 		cubeJ--;
			// 	if (pointSel.z + 5.0 < 0)
			// 		cubeK--;

			// 	if (cubeI >= 0 && cubeI < laserCloudWidth &&
			// 		cubeJ >= 0 && cubeJ < laserCloudHeight &&
			// 		cubeK >= 0 && cubeK < laserCloudDepth)
			// 	{
			// 		int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
			// 		laserCloudCornerArray[cubeInd]->push_back(pointSel);
			// 	}
			// }
			//非降采样
			for (int i = 0; i < laserCloudCornerLastNum; i++)
			{
				pointAssociateToMap(&laserCloudCornerLast->points[i], &pointSel);

				int cubeI = int((pointSel.x + 5.0) / 10.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 5.0) / 10.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 5.0) / 10.0) + laserCloudCenDepth;

				if (pointSel.x + 5.0 < 0)
					cubeI--;
				if (pointSel.y + 5.0 < 0)
					cubeJ--;
				if (pointSel.z + 5.0 < 0)
					cubeK--;

				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudCornerArray[cubeInd]->push_back(pointSel);
				}
			}


                // cout << "11" << endl;


			for (int i = 0; i < laserCloudSurfStackNum; i++)
			{
				pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

				int cubeI = int((pointSel.x + 5.0) / 10.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 5.0) / 10.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 5.0) / 10.0) + laserCloudCenDepth;

				if (pointSel.x + 5.0 < 0)
					cubeI--;
				if (pointSel.y + 5.0 < 0)
					cubeJ--;
				if (pointSel.z + 5.0 < 0)
					cubeK--;

				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudSurfArray[cubeInd]->push_back(pointSel);
				}
			}
			printf("add points time %f ms\n", t_add.toc());

			
			TicToc t_filter;

			for (int i = 0; i < laserCloudValidNum; i++)
			{
				int ind = laserCloudValidInd[i];

				pcl::PointCloud<PointType>::Ptr tmpCorner(new pcl::PointCloud<PointType>());
				downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
				downSizeFilterCorner.filter(*tmpCorner);
				laserCloudCornerArray[ind] = tmpCorner;

				pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
				downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
				downSizeFilterSurf.filter(*tmpSurf);
				laserCloudSurfArray[ind] = tmpSurf;
			}
			printf("filter time %f ms \n", t_filter.toc());
			
			TicToc t_pub;
			// publish surround map for every 5 frame
			if (frameCount % 5 == 0)
			{
				laserCloudSurround->clear();
				for (int i = 0; i < laserCloudSurroundNum; i++)
				{
					int ind = laserCloudSurroundInd[i];
					*laserCloudSurround += *laserCloudCornerArray[ind];
					*laserCloudSurround += *laserCloudSurfArray[ind];
				}

				sensor_msgs::PointCloud2 laserCloudSurround3;
				pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
				laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudSurround3.header.frame_id = "/camera_init";
				pubLaserCloudSurround.publish(laserCloudSurround3);
			}

			if (frameCount % 20 == 0)
			{
				pcl::PointCloud<PointType> laserCloudMap;
				for (int i = 0; i < 4851; i++)
				{
					laserCloudMap += *laserCloudCornerArray[i];
					laserCloudMap += *laserCloudSurfArray[i];
				}
				sensor_msgs::PointCloud2 laserCloudMsg;
				pcl::toROSMsg(laserCloudMap, laserCloudMsg);
				laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudMsg.header.frame_id = "/camera_init";
				pubLaserCloudMap.publish(laserCloudMsg);
			}

			int laserCloudFullResNum = laserCloudFullRes->points.size();
			for (int i = 0; i < laserCloudFullResNum; i++)
			{
				pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
			}

			sensor_msgs::PointCloud2 laserCloudFullRes3;
			pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
			laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			laserCloudFullRes3.header.frame_id = "/camera_init";
			pubLaserCloudFullRes.publish(laserCloudFullRes3);

			printf("mapping pub time %f ms \n", t_pub.toc());

			printf("whole mapping time %f ms +++++\n", t_whole.toc());

			nav_msgs::Odometry odomAftMapped;
			odomAftMapped.header.frame_id = "/camera_init";
			odomAftMapped.child_frame_id = "/aft_mapped";
			odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
			odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
			odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
			odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
			odomAftMapped.pose.pose.position.x = t_w_curr.x();
			odomAftMapped.pose.pose.position.y = t_w_curr.y();
			odomAftMapped.pose.pose.position.z = t_w_curr.z();
			pubOdomAftMapped.publish(odomAftMapped);

			geometry_msgs::PoseStamped laserAfterMappedPose;
			laserAfterMappedPose.header = odomAftMapped.header;
			laserAfterMappedPose.pose = odomAftMapped.pose.pose;
			laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
			laserAfterMappedPath.header.frame_id = "/camera_init";
			laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
			pubLaserAfterMappedPath.publish(laserAfterMappedPath);

			static tf::TransformBroadcaster br;
			tf::Transform transform;
			tf::Quaternion q;
			transform.setOrigin(tf::Vector3(t_w_curr(0),
											t_w_curr(1),
											t_w_curr(2)));
			q.setW(q_w_curr.w());
			q.setX(q_w_curr.x());
			q.setY(q_w_curr.y());
			q.setZ(q_w_curr.z());
			transform.setRotation(q);
			br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "/camera_init", "/aft_mapped"));
			// br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped"));


			frameCount++;
		}
		
        // while (!fullResBuf.empty())
        // {

        // }
        
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
	}
}
int skipFrameNum1 = 5;

int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserMapping");
	ros::NodeHandle nh;

	float lineRes = 0;
	float planeRes = 0;
	nh.param<float>("mapping_line_resolution", lineRes, 0.4);
	nh.param<float>("mapping_plane_resolution", planeRes, 0.8);
    nh.param<int>("map_layer", skipFrameNum1, 2);

	printf("line resolution %f plane resolution %f \n", lineRes, planeRes);
	downSizeFilterCorner.setLeafSize(lineRes, lineRes,lineRes);
	downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);
	map_cnt = skipFrameNum1;
	ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/surfel_map", 100, laserCloudCornerLastHandler);

	// ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);

	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);

	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/surfel_map", 100, laserCloudFullResHandler);

	pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);

	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);

	pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);

	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);

	pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);

	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

	for (int i = 0; i < laserCloudNum; i++)
	{
		laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
		laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
	}

	std::thread mapping_process{process};

	ros::spin();

	return 0;
}