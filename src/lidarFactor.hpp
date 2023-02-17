// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include <feature/lidar_surfel.h>

struct LidarSurfelsFactor
{
	LidarSurfelsFactor(clins::SurfelCorrespondence surfel_corr_, double s_)
		: surfel_corr(surfel_corr_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		
		Eigen::Matrix<T, 3, 1> cur_s{T(surfel_corr.cur_surfel_u[0]), T(surfel_corr.cur_surfel_u[1]), T(surfel_corr.cur_surfel_u[2])};
		Eigen::Matrix<T, 3, 1> map_s{T(surfel_corr.map_surfel_u[0]), T(surfel_corr.map_surfel_u[1]), T(surfel_corr.map_surfel_u[2])};
		Eigen::Matrix<T, 3, 1> normal_map{T(surfel_corr.normal_surfel[0]), T(surfel_corr.normal_surfel[1]), T(surfel_corr.normal_surfel[2])};
		T weight = T(surfel_corr.weight);


		// Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		// Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		// Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		// q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cur_s + t_last_curr;

		// Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		// Eigen::Matrix<T, 3, 1> de = lpa - lpb;

		// Eigen::Matrix<T, 1, 1> nu = (lp - map_s).dot(normal_map);

		residual[0] = weight * (lp - map_s).dot(normal_map);
		// residual[1] = nu.y() / de.norm();
		// residual[2] = nu.z() / de.norm();

		return true;
	}

	static ceres::CostFunction *Create(const clins::SurfelCorrespondence surfel_corr_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarSurfelsFactor, 1, 4, 3>(
			new LidarSurfelsFactor(surfel_corr_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	clins::SurfelCorrespondence surfel_corr;
	double s;
};

struct LidarSurfelsWithoutweightFactor
{
	LidarSurfelsWithoutweightFactor(clins::SurfelCorrespondence surfel_corr_, double s_)
		: surfel_corr(surfel_corr_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		
		Eigen::Matrix<T, 3, 1> cur_s{T(surfel_corr.cur_surfel_u[0]), T(surfel_corr.cur_surfel_u[1]), T(surfel_corr.cur_surfel_u[2])};
		Eigen::Matrix<T, 3, 1> map_s{T(surfel_corr.map_surfel_u[0]), T(surfel_corr.map_surfel_u[1]), T(surfel_corr.map_surfel_u[2])};
		Eigen::Matrix<T, 3, 1> normal_map{T(surfel_corr.normal_surfel[0]), T(surfel_corr.normal_surfel[1]), T(surfel_corr.normal_surfel[2])};
		// T weight = T(surfel_corr.weight);


		// Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		// Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		// Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		// q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cur_s + t_last_curr;

		// Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		// Eigen::Matrix<T, 3, 1> de = lpa - lpb;

		// Eigen::Matrix<T, 1, 1> nu = (lp - map_s).dot(normal_map);

		residual[0] = (lp - map_s).dot(normal_map);
		// residual[1] = nu.y() / de.norm();
		// residual[2] = nu.z() / de.norm();

		return true;
	}

	static ceres::CostFunction *Create(const clins::SurfelCorrespondence surfel_corr_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarSurfelsWithoutweightFactor, 1, 4, 3>(
			new LidarSurfelsWithoutweightFactor(surfel_corr_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	clins::SurfelCorrespondence surfel_corr;
	double s;
};



struct LidarSurfelsRotFactor
{
	LidarSurfelsRotFactor(clins::SurfelCorrespondence surfel_corr_, double s_)
		: surfel_corr(surfel_corr_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		
		Eigen::Matrix<T, 3, 1> normal_cur{T(surfel_corr.normal_cur[0]), T(surfel_corr.normal_cur[1]), T(surfel_corr.normal_cur[2])};
		// Eigen::Matrix<T, 3, 1> map_s{T(surfel_corr.map_surfel_u[0]), T(surfel_corr.map_surfel_u[1]), T(surfel_corr.map_surfel_u[2])};
		Eigen::Matrix<T, 3, 1> normal_map{T(surfel_corr.normal_surfel[0]), T(surfel_corr.normal_surfel[1]), T(surfel_corr.normal_surfel[2])};
		// T weight = T(surfel_corr.weight);

		// Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		// Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		// Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		// q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;

		lp = q_last_curr.toRotationMatrix() * normal_cur;
		// lp = q_last_curr * cur_s + t_last_curr;

		// Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		// Eigen::Matrix<T, 3, 1> de = lpa - lpb;

		// Eigen::Matrix<T, 1, 1> nu = (lp - map_s).dot(normal_map);

		residual[0] = lp.cross(normal_map).squaredNorm();
		// residual[1] = nu.y() / de.norm();
		// residual[2] = nu.z() / de.norm();

		return true;
	}

	static ceres::CostFunction *Create(const clins::SurfelCorrespondence surfel_corr_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarSurfelsRotFactor, 1, 4, 3>(
			new LidarSurfelsRotFactor(surfel_corr_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	clins::SurfelCorrespondence surfel_corr;
	double s;
};

struct LidarSurfelsSmothnessFactor
{
	LidarSurfelsSmothnessFactor(Eigen::Quaterniond q_last_smooth_, Eigen::Vector3d t_last_smooth_, double s_)
		: q_last_smooth(q_last_smooth_), t_last_smooth(t_last_smooth_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		
		// Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};

		Eigen::Matrix<T, 3, 1> t_last_1{T(t_last_smooth.x()), T(t_last_smooth.y()), T(t_last_smooth.z())};
		Eigen::Quaternion<T> q_last_1{T(q_last_smooth.w()), T(q_last_smooth.x()), T(q_last_smooth.y()), T(q_last_smooth.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		// q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};



		residual[0] = (t_last_1 - t_last_curr).norm();
		// residual[1] = nu.y() / de.norm();
		// residual[2] = nu.z() / de.norm();

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Quaterniond q_last_smooth_,const Eigen::Vector3d t_last_smooth_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarSurfelsSmothnessFactor, 1, 4, 3>(
			new LidarSurfelsSmothnessFactor(q_last_smooth_, t_last_smooth_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	clins::SurfelCorrespondence surfel_corr;
	Eigen::Quaterniond q_last_smooth;
	Eigen::Vector3d t_last_smooth;
	double s;
};


struct LidarEdgeFactor
{
	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double s_)
		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<T, 3, 1> de = lpa - lpb;

		residual[0] = nu.x() / de.norm();
		residual[1] = nu.y() / de.norm();
		residual[2] = nu.z() / de.norm();

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarEdgeFactor, 3, 4, 3>(
			new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_a, last_point_b;
	double s;
};

struct LidarPlaneFactor
{
	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
		  last_point_m(last_point_m_), s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
		ljm_norm.normalize();
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		//Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
		//Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		residual[0] = (lp - lpj).dot(ljm);

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									   const double s_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneFactor, 1, 4, 3>(
			new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
	}

	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
	double s;
};

struct LidarPlaneNormFactor
{

	LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
						 double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
														 negative_OA_dot_norm(negative_OA_dot_norm_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;

		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactor, 1, 4, 3>(
			new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};


struct LidarDistanceFactor
{

	LidarDistanceFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_) 
						: curr_point(curr_point_), closed_point(closed_point_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;


		residual[0] = point_w.x() - T(closed_point.x());
		residual[1] = point_w.y() - T(closed_point.y());
		residual[2] = point_w.z() - T(closed_point.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactor, 3, 4, 3>(
			new LidarDistanceFactor(curr_point_, closed_point_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d closed_point;
};