/*
 * @Description:cere优化文件
 * @Author: Wang Dongsheng, Sun Qinxuan
 * @Date: 2021-11-15 17:00:00
 */

#ifndef VSION_LOCALIZATION_MODELS_GRAPH_OPTIMIZER_CERES_CERES_GRAPH_OPTIMIZER_HPP_
#define VSION_LOCALIZATION_MODELS_GRAPH_OPTIMIZER_CERES_CERES_GRAPH_OPTIMIZER_HPP_

#include <ceres/ceres.h>
#include <sophus/se3.h>
#include <sophus/so3.h>

#include <vector>
#include <yaml-cpp/yaml.h>

#include "global_defination/message_print.hpp"

#include "models/graph_optimizer/ceres/types/edge_pose.hpp"
#include "models/graph_optimizer/ceres/types/vertex_pose.hpp"
#include "models/graph_optimizer/interface_graph_optimizer.hpp"

namespace vision_localization
{
class CeresGraphOptimizer : public InterfaceGraphOptimizer
{
public:
  using MapVertexPose = std::map<unsigned int, VertexPose>;
  using DequeEdgePose = std::deque<EdgePose>;

  CeresGraphOptimizer(const YAML::Node &node);

  bool Optimize(bool use_analytic_jacobian = false) override;

  bool GetOptimizedPose(std::deque<Eigen::Isometry3f> &optimized_pose) override;
  bool UpdateGraphNode() override;
  Eigen::Isometry3f GetOptimizedPose(int vertex_index) override;
  Eigen::Isometry3f GetOptimizedPose() override;
  Eigen::Matrix<float, 6, 6> GetOptimizedCovariance(int vertex_index) override;
  int GetNodeNum() override;

  void AddSe3Node(const Eigen::Isometry3f &pose, bool need_fix) override;
  void AddSe3Node(const Eigen::Isometry3f &pose, unsigned int vertex_index, bool need_fix) override;
  void RemoveSe3Node(unsigned int vertex_index) override;
  void RemoveSe3Node() override;
  void AddSe3Edge(int vertex_index1, int vertex_index2, const Eigen::Isometry3f &relative_pose,
                  const Eigen::MatrixXf covariance) override;
  void AddSe3PreintEdge(int vertex_index1, int vertex_index2, const Eigen::Isometry3f &relative_pose,
                        const Eigen::MatrixXf covariance) override;
  void AddSe3LoopEdge(int vertex_index1, int vertex_index2, const Eigen::Isometry3f &relative_pose,
                      const Eigen::MatrixXf covariance) override;
  void AddSe3PriorXYZEdge(int se3_vertex_index, const Eigen::Vector3f &xyz, Eigen::VectorXf noise) override;
  void AddSe3PriordZEdge(int vertex_index1, int vertex_index2, const double delta_z,
                         const Eigen::VectorXf noise) override;
  void AddSe3PriorZEdge(int vertex_index1, const Eigen::Vector2f height_range, const Eigen::VectorXf noise) override;
  void AddSe3PriordRPEdge(int vertex_index1, int vertex_index2, const Eigen::Vector2f delta_RP,
                          const Eigen::VectorXf noise) override;

private:
  bool SolveCeresProblem(ceres::Problem *problem);
  bool EstimateAndCullingWrongLoop(ceres::Problem *problem);
  void DisplayInformation();
  double EvaluateEdgeInformation(const DequeEdgePose &edges, double weight);

  void AddRobustKernel(std::string kernel_type, double kernel_size);
  Eigen::MatrixXd CalculateDiagMatrix(Eigen::VectorXd noise);
  Eigen::MatrixXd CalculateInformationMatrix(Eigen::MatrixXd cov);

  // [quaternion,vector] parameterization;
  // automatic derivative for Jacobian;
  bool BuildCeresProblem(ceres::Problem *problem);
  bool AddConstraints(ceres::Problem *problem);
  bool EstimateCovariance(ceres::Problem *problem);

  // using SE3 parameterization;
  // analytic derivative for Jacobian;
  bool BuildCeresProblemSE3(ceres::Problem *problem);
  bool EstimateCovarianceSE3(ceres::Problem *problem);

private:
  MapVertexPose vertices_;
  DequeEdgePose edges_;
  DequeEdgePose edges_preint_;
  DequeEdgePose edges_loop_;
  DequeEdgePose edges_gnss_;
  DequeEdgePose edges_rel_height_;
  DequeEdgePose edges_abs_height_;
  DequeEdgePose edges_roll_pitch_;

  bool ceres_debug_;

  double weight_preint_edge_;
  double weight_loop_edge_;
  double weight_gnss_edge_;

  std::vector<ceres::ResidualBlockId> loop_res_ids_;

  ceres::LossFunction *loss_function_ = nullptr;
  std::string kernel_type_ = "NONE";
  double kernel_size_;

  ceres::LinearSolverType linear_solver_type_ = ceres::SPARSE_NORMAL_CHOLESKY;
  ceres::TrustRegionStrategyType trust_region_strategy_type_ = ceres::LEVENBERG_MARQUARDT;
  unsigned int max_num_iterations_ = 10;
  bool minimizer_progress_to_stdout_ = false;
};
}  // namespace vision_localization
#endif
