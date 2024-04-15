/***********************************************
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified:	2021-11-11 10:47
#
# Filename:		vertex_pose.hpp
#
# Description:
#
************************************************/

#ifndef VSION_LOCALIZATION_MODELS_GRAPH_OPTIMIZER_VERTEX_POSE_HPP_
#define VSION_LOCALIZATION_MODELS_GRAPH_OPTIMIZER_VERTEX_POSE_HPP_

#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace vision_localization
{
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

class VertexPose
{
public:
  VertexPose()
  {
  }
  VertexPose(unsigned int i, const Eigen::Isometry3d &pose, bool fix) : id(i), fixed(fix)
  {
    position = pose.translation();
    quaternion = Eigen::Quaterniond(pose.linear());
    quaternion.normalize();
    xi = Sophus::SE3(quaternion.normalized(), position).log();
  }

  unsigned int id;
  Eigen::Vector3d position;
  Eigen::Quaterniond quaternion;
  Matrix6d covariance;

  Vector6d xi;

  bool fixed;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class SE3Parameterization : public ceres::LocalParameterization
{
public:
  SE3Parameterization()
  {
  }
  virtual ~SE3Parameterization()
  {
  }

  virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const
  {
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(x);
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_lie(delta);

    Sophus::SE3 T = Sophus::SE3::exp(lie);
    Sophus::SE3 delta_T = Sophus::SE3::exp(delta_lie);

    Eigen::Matrix<double, 6, 1> x_plus_delta_lie = (delta_T * T).log();

    for (int i = 0; i < 6; ++i) {
      x_plus_delta[i] = x_plus_delta_lie(i, 0);
    }

    return true;
  }
  virtual bool ComputeJacobian(const double *x, double *jacobian) const
  {
    ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
    return true;
  }
  virtual int GlobalSize() const
  {
    return 6;
  }
  virtual int LocalSize() const
  {
    return 6;
  }
};

}  // namespace vision_localization
#endif
