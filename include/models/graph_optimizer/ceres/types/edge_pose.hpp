/***********************************************
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified:	2021-11-11 09:32
#
# Filename:		edge_pose.hpp
#
# Description:
#
************************************************/

#ifndef VSION_LOCALIZATION_MODELS_GRAPH_OPTIMIZER_EDGE_POSE_HPP_
#define VSION_LOCALIZATION_MODELS_GRAPH_OPTIMIZER_EDGE_POSE_HPP_

#include "models/graph_optimizer/ceres/types/vertex_pose.hpp"
#include "tools/convert_matrix.hpp"
#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace vision_localization
{
class EdgePose
{
public:
  EdgePose(unsigned int i, unsigned int j, const Eigen::Isometry3d &delta_pose, const Matrix6d &info)
      : id_i(i), id_j(j), information(info)  //, pose_ij(delta_pose)
  {
    Eigen::Quaterniond q(delta_pose.linear());
    q.normalize();
    pose_ij = delta_pose;
    pose_ij.linear() = q.toRotationMatrix();
  }

  unsigned int id_i, id_j;

  // transformation representing the pose of frame j w.r.t. the frame i;
  Eigen::Isometry3d pose_ij;

  // inverse of the covariance matrix;
  // order: [x,y,z,rx,ry,rz]^T;
  Matrix6d information;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class EdgePoseError
{
public:
  EdgePoseError(const Eigen::Isometry3d &delta_ij, const Matrix6d &sqrt_information)
      : pose_ij_meas_(delta_ij), sqrt_information_(sqrt_information)
  {
  }

  // pose - [x,y,z,qx,qy,qz,qw]^T
  // residual_ptr - [x,y,z,rx,ry,rz]^T;
  template <typename T>
  bool operator()(const T *const position_i, const T *const orientation_i, const T *const position_j,
                  const T *const orientation_j, T *residual_ptr) const
  {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_i(position_i);
    Eigen::Map<const Eigen::Quaternion<T>> q_i(orientation_i);

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_j(position_j);
    Eigen::Map<const Eigen::Quaternion<T>> q_j(orientation_j);

    Eigen::Quaternion<T> q_i_inv = q_i.conjugate();
    Eigen::Quaternion<T> q_ij = q_i_inv * q_j;
    Eigen::Matrix<T, 3, 1> p_ij = q_i_inv * (p_j - p_i);

    Eigen::Quaternion<T> q_ij_meas(pose_ij_meas_.linear().template cast<T>());
    Eigen::Quaternion<T> delta_q = q_ij_meas * q_ij.conjugate();

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residual_ptr);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_trs(residual_ptr);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_rot(residual_ptr + 3);

    residual_trs = p_ij - pose_ij_meas_.translation().template cast<T>();
    residual_rot = T(2.0) * delta_q.vec();
    residual.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Isometry3d &pose_ij, const Matrix6d &sqrt_information)
  {
    return new ceres::AutoDiffCostFunction<EdgePoseError, 6, 3, 4, 3, 4>(new EdgePoseError(pose_ij, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  Eigen::Isometry3d pose_ij_meas_;
  Matrix6d sqrt_information_;
};

class EdgeGNSSError
{
public:
  EdgeGNSSError(const Eigen::Vector3d &position, const Eigen::Matrix3d &sqrt_information)
      : position_gnss_(position), sqrt_information_(sqrt_information)
  {
  }

  // position - [x,y,z]^T
  // residual_ptr - [x,y,z]^T;
  template <typename T> bool operator()(const T *const position, T *residual_ptr) const
  {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> pos(position);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(residual_ptr);

    residual = position_gnss_.template cast<T>() - pos;
    residual.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d &position, const Eigen::Matrix3d &sqrt_information)
  {
    return new ceres::AutoDiffCostFunction<EdgeGNSSError, 3, 3>(new EdgeGNSSError(position, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  Eigen::Vector3d position_gnss_;
  Eigen::Matrix3d sqrt_information_;
};

class EdgeRelHeightError
{
public:
  EdgeRelHeightError(const double &delta_height, const double &sqrt_information)
      : delta_height_ij_(delta_height), sqrt_information_(sqrt_information)
  {
  }

  // pose - [x,y,z,qx,qy,qz,qw]^T
  // residual_ptr - delta_z;
  template <typename T>
  bool operator()(const T *const position_i, const T *const orientation_i, const T *const position_j,
                  const T *const orientation_j, T *residual_ptr) const
  {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_i(position_i);
    Eigen::Map<const Eigen::Quaternion<T>> q_i(orientation_i);

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_j(position_j);
    Eigen::Map<const Eigen::Quaternion<T>> q_j(orientation_j);

    Eigen::Quaternion<T> q_i_inv = q_i.conjugate();
    Eigen::Matrix<T, 3, 1> p_ij = q_i_inv * (p_j - p_i);
    *residual_ptr = T(sqrt_information_) * (T(delta_height_ij_) - p_ij(2));

    return true;
  }

  static ceres::CostFunction *Create(const double &delta_height, const double &sqrt_information)
  {
    return new ceres::AutoDiffCostFunction<EdgeRelHeightError, 1, 3, 4, 3, 4>(
        new EdgeRelHeightError(delta_height, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  double delta_height_ij_;
  double sqrt_information_;
};

class EdgeAbsHeightError
{
public:
  EdgeAbsHeightError(const double &height, const double &sqrt_information)
      : height_(height), sqrt_information_(sqrt_information)
  {
  }

  template <typename T> bool operator()(const T *const position, T *residual_ptr) const
  {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> pos(position);
    *residual_ptr = T(sqrt_information_) * (T(height_) - pos(2));
    return true;
  }

  static ceres::CostFunction *Create(const double &height, const double &sqrt_information)
  {
    return new ceres::AutoDiffCostFunction<EdgeAbsHeightError, 1, 3>(new EdgeAbsHeightError(height, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  double height_;
  double sqrt_information_;
};

class EdgeRollPitchError
{
public:
  EdgeRollPitchError(const Eigen::Vector2d &rp, const Eigen::Matrix2d &sqrt_information)
      : roll_pitch_(rp), sqrt_information_(sqrt_information)
  {
  }

  template <typename T>
  bool operator()(const T *const orientation_i, const T *const orientation_j, T *residual_ptr) const
  {
    Eigen::Map<const Eigen::Quaternion<T>> q_i(orientation_i);
    Eigen::Map<const Eigen::Quaternion<T>> q_j(orientation_j);

    Eigen::Quaternion<T> q_ij = q_i.conjugate() * q_j;
    Eigen::Matrix<T, 3, 1> euler = toEulerAngle(q_ij);
    Eigen::Matrix<T, 2, 1> rp;
    rp(0) = euler(0);
    rp(1) = euler(1);

    Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(residual_ptr);
    residual = roll_pitch_.template cast<T>() - rp;  // euler.block<2,1>(0,0);
    residual.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &rp, const Eigen::Matrix2d &sqrt_information)
  {
    return new ceres::AutoDiffCostFunction<EdgeRollPitchError, 2, 4, 4>(new EdgeRollPitchError(rp, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  Eigen::Vector2d roll_pitch_;
  Eigen::Matrix2d sqrt_information_;

  template <typename T> Eigen::Matrix<T, 3, 1> toEulerAngle(const Eigen::Quaternion<T> &q) const
  {
    Eigen::Matrix<T, 3, 1> rpy;
    // roll (x-axis rotation)
    T sinr_cosp = +T(2.0) * (q.w() * q.x() + q.y() * q.z());
    T cosr_cosp = +T(1.0) - T(2.0) * (q.x() * q.x() + q.y() * q.y());
    rpy(0) = atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    T sinp = +T(2.0) * (q.w() * q.y() - q.z() * q.x());
    if (abs(sinp) >= T(1.0)) {
      if (sinp < T(0.0))
        rpy(1) = T(-M_PI / 2.0);
      else
        rpy(1) = T(M_PI / 2.0);
    } else
      rpy(1) = asin(sinp);

    // yaw (z-axis rotation)
    T siny_cosp = +T(2.0) * (q.w() * q.z() + q.x() * q.y());
    T cosy_cosp = +T(1.0) - T(2.0) * (q.y() * q.y() + q.z() * q.z());
    rpy(2) = atan2(siny_cosp, cosy_cosp);

    return rpy;
  }
};

class EdgePoseSE3CostFunction : public ceres::SizedCostFunction<6, 6, 6>
{
public:
  ~EdgePoseSE3CostFunction()
  {
  }
  EdgePoseSE3CostFunction(Sophus::SE3 meas, Matrix6d sqrt_info) : measurment_(meas), sqrt_information_(sqrt_info)
  {
  }

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
  {
    Sophus::SE3 pose_i = Sophus::SE3::exp(Vector6d(parameters[0]));
    Sophus::SE3 pose_j = Sophus::SE3::exp(Vector6d(parameters[1]));
    Sophus::SE3 estimate = pose_i.inverse() * pose_j;

    Eigen::Map<Vector6d> residual(residuals);
    residual = sqrt_information_ * ((measurment_.inverse() * estimate).log());

    if (jacobians) {
      if (jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jacobian_i(jacobians[0]);
        Matrix6d J = JRInv(Sophus::SE3::exp(residual));
        jacobian_i = (-J) * pose_j.inverse().Adj();
        jacobian_i = sqrt_information_ * (-J) * pose_j.inverse().Adj();
      }
      if (jacobians[1]) {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jacobian_j(jacobians[1]);
        Matrix6d J = JRInv(Sophus::SE3::exp(residual));
        jacobian_j = J * pose_j.inverse().Adj();
        jacobian_j = sqrt_information_ * J * pose_j.inverse().Adj();
      }
    }
    return true;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

private:
  const Sophus::SE3 measurment_;
  const Matrix6d sqrt_information_;

  Matrix6d JRInv(const Sophus::SE3 &e) const
  {
    Matrix6d J;
    J.block(0, 0, 3, 3) = Sophus::SO3::hat(e.so3().log());
    J.block(0, 3, 3, 3) = Sophus::SO3::hat(e.translation());
    J.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero(3, 3);
    J.block(3, 3, 3, 3) = Sophus::SO3::hat(e.so3().log());
    J = 0.5 * J + Matrix6d::Identity();
    return J;
  }
};

}  // namespace vision_localization
#endif
