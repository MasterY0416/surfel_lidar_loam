//
// Created by dell on 22-8-10.
//

#ifndef SRC_LIDAR_SURFEL_H
#define SRC_LIDAR_SURFEL_H

#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <lidar_data.h>
#include <voxel_util.hpp>
#include <glog/logging.h>
namespace clins {
    struct LiDARSurfel{
        LiDARSurfel(){
            timestamp = 0;
            time_max = 0;
            surfel_vec_ = std::make_shared<std::vector<VoxelShape*>>();
        }

        LiDARSurfel(const LiDARSurfel &fea){
            timestamp = fea.timestamp;
            time_max = fea.time_max;
            surfel_vec_ = std::make_shared<std::vector<VoxelShape*>>();
            surfel_vec_->resize(fea.surfel_vec_->size());
            for(int i = 0;i < fea.surfel_vec_->size();i++){
                CHECK(fea.surfel_vec_->at(i)) << "fea.surfel_vec_->at(i) == nullptr";
                VoxelShape *temp = new VoxelShape(*(fea.surfel_vec_->at(i)));
                surfel_vec_->at(i) = temp;
            }
            //surfel_vec_->resize(fea.surfel_vec_->size());
            //surfel_vec_ = *fea.surfel_vec_;
        }

        LiDARSurfel &operator=(const LiDARSurfel &fea){
            if(this != &fea){
                LiDARSurfel temp(fea);
                this->timestamp = temp.timestamp;
                this->time_max = temp.time_max;
                //TODO: is this swap need ?
                std::shared_ptr<std::vector<VoxelShape*>> p_temp = temp.surfel_vec_;
                temp.surfel_vec_ = this->surfel_vec_;
                this->surfel_vec_ = p_temp;
            }

            return *this;
        }

        void Clear(){
            timestamp = 0;
            time_max = 0;
            for(auto it = surfel_vec_->begin();it != surfel_vec_->end();it++) {
                delete *it;
            }
            surfel_vec_->clear();


        }

        double timestamp;
        double time_max;
        std::shared_ptr<std::vector<VoxelShape*>>  surfel_vec_;
        std::shared_ptr<std::deque<VoxelShape*>>  surfel_deque_;

    };

    struct SurfelCorrespondence {
        double t_surfel;
        double t_map;

        Eigen::Vector3d cur_surfel_u;
        Eigen::Vector3d map_surfel_u;
        Eigen::Vector3d normal_surfel;
        Eigen::Vector3d normal_cur;
        float weight;
        //TODO: need lambad1 ? or just add in normal
        // need contain two voxelshape pair
        // add need finish plane to plane resdiual
        // and normal must compute from two voxel
    };
}

#endif //SRC_LIDAR_SURFEL_H
