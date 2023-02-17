#ifndef POINT_DATA_H
#define POINT_DATA_H

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>

namespace point_data {
struct EIGEN_ALIGN16 PointcloudXYZITR {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PCL_ADD_POINT4D;
  float intensity;
  double timestamp;
  uint16_t ring;
  static inline PointcloudXYZITR make(float x, float y, float z,
                                      float intensity, double timestamp, uint16_t ring) {
    return {x, y, z, 0.0f, intensity, timestamp, ring};
  }
};

} // namespace point_data

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT( point_data::PointcloudXYZITR,
   (float, x,x)
   (float, y, y)
   (float, z, z)
   (float, intensity, intensity)
   (double, timestamp, timestamp)
   (uint16_t, ring, ring)
)
// clang-format on
#endif // POINT_DATA_H
