/*
 * @Description: ceres优化求解器实现
 * @Author: Wang Dongsheng, Sun Qinxuan
 * @Date: 2021-11-15 17:00:00
 */

#include "models/graph_optimizer/ceres/ceres_graph_optimizer.hpp"
#include "global_defination/message_print.hpp"
#include "tools/tic_toc.hpp"

namespace vision_localization
{
CeresGraphOptimizer::CeresGraphOptimizer(const YAML::Node &node)
{
  if (node["solver_type"].as<std::string>() == "sparse_normal_cholesky") {
    linear_solver_type_ = ceres::SPARSE_NORMAL_CHOLESKY;
  }
  if (node["solver_type"].as<std::string>() == "dense_normal_cholesky") {
    linear_solver_type_ = ceres::DENSE_NORMAL_CHOLESKY;
  }
  if (node["solver_type"].as<std::string>() == "sparse_schur") {
    linear_solver_type_ = ceres::SPARSE_SCHUR;
  }
  if (node["solver_type"].as<std::string>() == "dense_schur") {
    linear_solver_type_ = ceres::DENSE_SCHUR;
  }
  if (node["trust_region_type"].as<std::string>() == "LM") {
    trust_region_strategy_type_ = ceres::LEVENBERG_MARQUARDT;
  }
  if (node["trust_region_type"].as<std::string>() == "DogLeg") {
    trust_region_strategy_type_ = ceres::DOGLEG;
  }
  max_num_iterations_ = node["max_num_iterations"].as<int>();
  minimizer_progress_to_stdout_ = node["minimizer_progress_to_stdout"].as<bool>();
  kernel_type_ = node["kernel_type"].as<std::string>();
  kernel_size_ = node["kernel_size"].as<double>();
  ceres_debug_ = node["ceres_debug"].as<bool>();
  if (node["use_preint_restrain"].as<bool>()) weight_preint_edge_ = node["weight_preint_edge"].as<double>();
  if (node["use_loop_close"].as<bool>()) weight_loop_edge_ = node["weight_loop_edge"].as<double>();
  if (node["use_gnss"].as<bool>()) weight_gnss_edge_ = node["weight_gnss_edge"].as<double>();
}
bool CeresGraphOptimizer::Optimize(bool use_analytic_jacobian)
{
  ceres::Problem problem;
  AddRobustKernel(kernel_type_, kernel_size_);

  if (use_analytic_jacobian) {
    if (!BuildCeresProblemSE3(&problem)) {
      ERROR("[ceres] building ceres problem failure!");
      return false;
    }
    if (!SolveCeresProblem(&problem)) {
      ERROR("[ceres] solving ceres problem failure!");
      return false;
    }
    // EstimateCovarianceSE3(&problem);
  } else {
    if (!BuildCeresProblem(&problem)) {
      ERROR("[ceres] building ceres problem failure!");
      return false;
    }
    AddConstraints(&problem);
    if (!SolveCeresProblem(&problem)) {
      ERROR("[backend] solving ceres problem failure!");
      return false;
    }

    if (EstimateAndCullingWrongLoop(&problem)) {
      if (!SolveCeresProblem(&problem)) {
        ERROR("[backend] solving ceres problem failure!");
        return false;
      }
    }
    // EstimateCovariance(&problem);
  }
  if (ceres_debug_) DisplayInformation();

  return true;
}
bool CeresGraphOptimizer::EstimateAndCullingWrongLoop(ceres::Problem *problem)
{
  if (loop_res_ids_.size() == 0) return false;

  ceres::Problem::EvaluateOptions EvalOpts;
  EvalOpts.num_threads = 1;
  // allows the user to switch the application of the loss function on and off.
  EvalOpts.apply_loss_function = false;
  EvalOpts.residual_blocks = loop_res_ids_;

  std::vector<double> Residuals;
  std::vector<double> Loss;
  double cost;
  problem->Evaluate(EvalOpts, &cost, &Residuals, nullptr, nullptr);

  if (Residuals.size() != loop_res_ids_.size() * 6) return false;

  double lost_acc = 0.0;
  for (std::size_t i = 0; i < Residuals.size(); i += 6) {
    double loss = 0.0;
    for (int j = 0; j < 6; j++) {
      loss = Residuals[i + j] * Residuals[i + j];
      lost_acc += loss;
    }
    Loss.push_back(loss);
  }

  double max = 0;
  double min = std::numeric_limits<double>::max();
  double mean = 0;
  for (std::size_t i = 0; i < Loss.size(); i++) {
    if (Loss[i] > max) max = Loss[i];
    if (Loss[i] < min) min = Loss[i];
    mean += Loss[i];
  }
  mean /= Loss.size();

  // remove loop constraint outliers
  double loss_thresh = 0.8 * mean + 0.2 * min;
  int bad_case_num = 0;
  for (std::size_t i = 0; i < Loss.size(); i++) {
    if (Loss[i] > loss_thresh || Loss[i] > 10.6446) {
      problem->RemoveResidualBlock(loop_res_ids_[i]);
      // INFO("(Loop index , loss value) : ",i,loop_res_ids_[i]);
      bad_case_num++;
    }
  }
  WARNING("[CeresGraphOptimizer][EstimateAndCullingWrongLoop] Find ", Loss.size() - bad_case_num, " inliers!");
  WARNING("[CeresGraphOptimizer][EstimateAndCullingWrongLoop] Find ", bad_case_num, " outliers!");

  if (bad_case_num > 0) return true;

  return false;
}
bool CeresGraphOptimizer::BuildCeresProblem(ceres::Problem *problem)
{
  if (problem == NULL) {
    ERROR("[backend][build problem] invalid optimization problem!");
    return false;
  }

  if (vertices_.size() == 0 || edges_.size() == 0) {
    ERROR("[ceres][build problem] empty pose graph!");
    return false;
  }

  ceres::LocalParameterization *quat_param = new ceres::EigenQuaternionParameterization;

  for (DequeEdgePose::const_iterator it = edges_.begin(); it != edges_.end(); ++it) {
    const EdgePose &edge = *it;

    MapVertexPose::iterator it_vertex_i = vertices_.find(edge.id_i);
    if (it_vertex_i == vertices_.end()) {
      edges_.erase(it);
      // WARNING("[ceres][odom_constraint] vertex with id ", edge.id_i, "not found!");
      continue;
    }

    MapVertexPose::iterator it_vertex_j = vertices_.find(edge.id_j);
    if (it_vertex_j == vertices_.end()) {
      edges_.erase(it);
      // WARNING("[ceres][odom_constraint] vertex with id ", edge.id_j, "not found!");
      continue;
    }

    if (it_vertex_i->second.fixed && it_vertex_j->second.fixed) {
      edges_.erase(it);
      continue;
    }

    const Matrix6d sqrt_info = edge.information.llt().matrixL();
    ceres::CostFunction *cost_function = EdgePoseError::Create(edge.pose_ij, sqrt_info);

    problem->AddResidualBlock(cost_function, loss_function_, it_vertex_i->second.position.data(),
                              it_vertex_i->second.quaternion.coeffs().data(), it_vertex_j->second.position.data(),
                              it_vertex_j->second.quaternion.coeffs().data());

    problem->SetParameterization(it_vertex_i->second.quaternion.coeffs().data(), quat_param);
    problem->SetParameterization(it_vertex_j->second.quaternion.coeffs().data(), quat_param);

    if (it_vertex_i->second.fixed) {
      problem->SetParameterBlockConstant(it_vertex_i->second.position.data());
      problem->SetParameterBlockConstant(it_vertex_i->second.quaternion.coeffs().data());
    }
    if (it_vertex_j->second.fixed) {
      problem->SetParameterBlockConstant(it_vertex_j->second.position.data());
      problem->SetParameterBlockConstant(it_vertex_j->second.quaternion.coeffs().data());
    }
  }

  for (DequeEdgePose::const_iterator it = edges_preint_.begin(); it != edges_preint_.end(); ++it) {
    const EdgePose &edge = *it;
    MapVertexPose::iterator it_vertex_i = vertices_.find(edge.id_i);
    if (it_vertex_i == vertices_.end()) {
      edges_preint_.erase(it);
      // WARNING("[backend][odom_constraint] vertex with id ", edge.id_i, "not found!");
      continue;
    }

    MapVertexPose::iterator it_vertex_j = vertices_.find(edge.id_j);
    if (it_vertex_j == vertices_.end()) {
      edges_preint_.erase(it);
      // WARNING("[backend][odom_constraint] vertex with id ", edge.id_j, "not found!");
      continue;
    }

    if (it_vertex_i->second.fixed && it_vertex_j->second.fixed) {
      edges_preint_.erase(it);
      continue;
    }

    Matrix6d sqrt_info = edge.information.llt().matrixL();
    sqrt_info *= weight_preint_edge_;
    ceres::CostFunction *cost_function = EdgePoseError::Create(edge.pose_ij, sqrt_info);

    problem->AddResidualBlock(cost_function, loss_function_, it_vertex_i->second.position.data(),
                              it_vertex_i->second.quaternion.coeffs().data(), it_vertex_j->second.position.data(),
                              it_vertex_j->second.quaternion.coeffs().data());

    problem->SetParameterization(it_vertex_i->second.quaternion.coeffs().data(), quat_param);
    problem->SetParameterization(it_vertex_j->second.quaternion.coeffs().data(), quat_param);

    if (it_vertex_i->second.fixed) {
      problem->SetParameterBlockConstant(it_vertex_i->second.position.data());
      problem->SetParameterBlockConstant(it_vertex_i->second.quaternion.coeffs().data());
    }
    if (it_vertex_j->second.fixed) {
      problem->SetParameterBlockConstant(it_vertex_j->second.position.data());
      problem->SetParameterBlockConstant(it_vertex_j->second.quaternion.coeffs().data());
    }
  }

  return true;
}

bool CeresGraphOptimizer::BuildCeresProblemSE3(ceres::Problem *problem)
{
  if (problem == nullptr) {
    ERROR("[ceres][build problem] invalid optimization problem!");
    return false;
  }

  if (vertices_.size() == 0 || edges_.size() == 0) {
    ERROR("[ceres][build problem] empty pose graph!");
    return false;
  }

  ceres::LocalParameterization *local_param = new SE3Parameterization();

  for (DequeEdgePose::const_iterator it = edges_.begin(); it != edges_.end(); ++it) {
    const EdgePose &edge = *it;

    MapVertexPose::iterator it_vertex_i = vertices_.find(edge.id_i);
    if (it_vertex_i == vertices_.end()) {
      // WARNING("[ceres][odom_constraint] vertex with id ", edge.id_i, "not found!");
      continue;
    }

    MapVertexPose::iterator it_vertex_j = vertices_.find(edge.id_j);
    if (it_vertex_j == vertices_.end()) {
      // WARNING("[ceres][odom_constraint] vertex with id ", edge.id_j, "not found!");
      continue;
    }

    Matrix6d sqrt_info = edge.information.llt().matrixL();
    Eigen::Quaterniond edge_q(edge.pose_ij.linear());
    Sophus::SE3 edge_se3(edge_q.normalized(), edge.pose_ij.translation());
    ceres::CostFunction *cost_function = new EdgePoseSE3CostFunction(edge_se3, sqrt_info);

    problem->AddResidualBlock(cost_function, loss_function_, it_vertex_i->second.xi.data(),
                              it_vertex_j->second.xi.data());

    problem->SetParameterization(it_vertex_i->second.xi.data(), local_param);
    problem->SetParameterization(it_vertex_j->second.xi.data(), local_param);

    if (it_vertex_i->second.fixed) {
      problem->SetParameterBlockConstant(it_vertex_i->second.xi.data());
    }
    if (it_vertex_j->second.fixed) {
      problem->SetParameterBlockConstant(it_vertex_j->second.xi.data());
    }
  }

  return true;
}

bool CeresGraphOptimizer::AddConstraints(ceres::Problem *problem)
{
  if (problem == nullptr) {
    ERROR("[ceres][additional constatints] invalid optimization problem!");
    return false;
  }

  if (vertices_.size() == 0 || edges_.size() + edges_preint_.size() == 0) {
    ERROR("[ceres][additional constatints] empty pose graph!");
    return false;
  }

  if (edges_gnss_.size() == 0 && edges_roll_pitch_.size() == 0 && edges_rel_height_.size() == 0 &&
      edges_abs_height_.size() == 0 && edges_loop_.size() == 0) {
    return false;
  }

  if (edges_loop_.size() != 0) {
    loop_res_ids_.clear();
    loop_res_ids_.reserve(edges_loop_.size());
  }
  for (DequeEdgePose::const_iterator it = edges_loop_.begin(); it != edges_loop_.end(); ++it) {
    const EdgePose &edge = *it;
    MapVertexPose::iterator it_vertex_i = vertices_.find(edge.id_i);
    if (it_vertex_i == vertices_.end()) {
      // WARNING("[backend][odom_constraint] vertex with id ", edge.id_i, "not found!");
      continue;
    }

    MapVertexPose::iterator it_vertex_j = vertices_.find(edge.id_j);
    if (it_vertex_j == vertices_.end()) {
      // WARNING("[backend][odom_constraint] vertex with id ", edge.id_j, "not found!");
      continue;
    }

    if (it_vertex_i->second.fixed && it_vertex_j->second.fixed) {
      edges_loop_.erase(it);
      continue;
    }

    Matrix6d sqrt_info = edge.information.llt().matrixL();
    sqrt_info *= weight_loop_edge_;
    ceres::CostFunction *cost_function = EdgePoseError::Create(edge.pose_ij, sqrt_info);

    auto loop_id =
        problem->AddResidualBlock(cost_function, loss_function_, it_vertex_i->second.position.data(),
                                  it_vertex_i->second.quaternion.coeffs().data(), it_vertex_j->second.position.data(),
                                  it_vertex_j->second.quaternion.coeffs().data());
    loop_res_ids_.push_back(loop_id);
  }


  for (DequeEdgePose::const_iterator it = edges_gnss_.begin(); it != edges_gnss_.end(); ++it) {
    const EdgePose &edge = *it;
    MapVertexPose::iterator it_vertex = vertices_.find(edge.id_i);
    if (it_vertex == vertices_.end()) {
      edges_gnss_.erase(it);
      // WARNING("[ceres][gnss_constraint] vertex with id ", edge.id_i, "not found!");
      continue;
    }

    if (it_vertex->second.fixed) {
      edges_gnss_.erase(it);
      continue;
    }

    Eigen::Matrix3d sqrt_info = edge.information.block<3, 3>(0, 0).llt().matrixL();
    sqrt_info *= weight_gnss_edge_;
    ceres::CostFunction *cost_function = EdgeGNSSError::Create(edge.pose_ij.translation(), sqrt_info);

    problem->AddResidualBlock(cost_function, loss_function_, it_vertex->second.position.data());
  }

  for (DequeEdgePose::const_iterator it = edges_rel_height_.begin(); it != edges_rel_height_.end(); ++it) {
    const EdgePose &edge = *it;
    MapVertexPose::iterator it_vertex_i = vertices_.find(edge.id_i);
    if (it_vertex_i == vertices_.end()) {
      edges_rel_height_.erase(it);
      // WARNING("[ceres][rel_height_constraint] vertex with id ", edge.id_i, "not found!");
      continue;
    }
    MapVertexPose::iterator it_vertex_j = vertices_.find(edge.id_j);
    if (it_vertex_j == vertices_.end()) {
      edges_rel_height_.erase(it);
      // WARNING("[ceres][rel_height_constraint] vertex with id ", edge.id_j, "not found!");
      continue;
    }

    if (it_vertex_i->second.fixed && it_vertex_j->second.fixed) {
      edges_rel_height_.erase(it);
      continue;
    }

    const double sqrt_info = sqrt(edge.information(2));
    ceres::CostFunction *cost_function = EdgeRelHeightError::Create(edge.pose_ij.translation()(2), sqrt_info);

    problem->AddResidualBlock(cost_function, loss_function_, it_vertex_i->second.position.data(),
                              it_vertex_i->second.quaternion.coeffs().data(), it_vertex_j->second.position.data(),
                              it_vertex_j->second.quaternion.coeffs().data());
  }

  for (DequeEdgePose::const_iterator it = edges_abs_height_.begin(); it != edges_abs_height_.end(); ++it) {
    const EdgePose &edge = *it;
    MapVertexPose::iterator it_vertex = vertices_.find(edge.id_i);
    if (it_vertex == vertices_.end()) {
      edges_abs_height_.erase(it);
      // WARNING("[ceres][abs_height_constraint] vertex with id ", edge.id_i, "not found!");
      continue;
    }

    if (it_vertex->second.fixed) {
      edges_abs_height_.erase(it);
      continue;
    }

    const double sqrt_info = sqrt(edge.information(2));
    ceres::CostFunction *cost_function = EdgeAbsHeightError::Create(edge.pose_ij.translation()(2), sqrt_info);

    problem->AddResidualBlock(cost_function, loss_function_, it_vertex->second.position.data());
  }

  for (DequeEdgePose::const_iterator it = edges_roll_pitch_.begin(); it != edges_roll_pitch_.end(); ++it) {
    const EdgePose &edge = *it;
    MapVertexPose::iterator it_vertex_i = vertices_.find(edge.id_i);
    if (it_vertex_i == vertices_.end()) {
      edges_roll_pitch_.erase(it);
      // WARNING("[ceres][roll_pitch_constraint] vertex with id ", edge.id_i, "not found!");
      continue;
    }
    MapVertexPose::iterator it_vertex_j = vertices_.find(edge.id_j);
    if (it_vertex_j == vertices_.end()) {
      edges_roll_pitch_.erase(it);
      // WARNING("[ceres][roll_pitch_constraint] vertex with id ", edge.id_j, "not found!");
      continue;
    }

    if (it_vertex_i->second.fixed && it_vertex_j->second.fixed) {
      edges_roll_pitch_.erase(it);
      continue;
    }

    const Eigen::Matrix2d sqrt_info = edge.information.block<2, 2>(0, 0).llt().matrixL();
    ceres::CostFunction *cost_function =
        EdgeRollPitchError::Create(edge.pose_ij.translation().block<2, 1>(0, 0), sqrt_info);

    problem->AddResidualBlock(cost_function, loss_function_, it_vertex_i->second.quaternion.coeffs().data(),
                              it_vertex_j->second.quaternion.coeffs().data());
  }

  return true;
}

bool CeresGraphOptimizer::SolveCeresProblem(ceres::Problem *problem)
{
  if (problem == nullptr) {
    ERROR("invalid optimization problem!");
    return false;
  }

  ceres::Solver::Summary summary;
  ceres::Solver::Options options;
  options.max_num_iterations = max_num_iterations_;
  options.linear_solver_type = linear_solver_type_;
  options.trust_region_strategy_type = trust_region_strategy_type_;
  options.minimizer_progress_to_stdout = minimizer_progress_to_stdout_;

  ceres::Solve(options, problem, &summary);
  if (minimizer_progress_to_stdout_) std::cout << summary.FullReport() << std::endl;

  return summary.IsSolutionUsable();
}

bool CeresGraphOptimizer::EstimateCovariance(ceres::Problem *problem)
{
  ceres::Covariance::Options options;
  ceres::Covariance covariance(options);

  std::vector<std::pair<const double *, const double *>> covariance_blocks;
  for (MapVertexPose::const_iterator it = vertices_.begin(); it != vertices_.end(); ++it) {
    const double *p = it->second.position.data();
    const double *q = it->second.quaternion.coeffs().data();
    covariance_blocks.push_back(std::make_pair(p, p));
    covariance_blocks.push_back(std::make_pair(q, q));
    covariance_blocks.push_back(std::make_pair(p, q));
  }

  CHECK(covariance.Compute(covariance_blocks, problem));

  for (MapVertexPose::iterator it = vertices_.begin(); it != vertices_.end(); ++it) {
    const double *p = it->second.position.data();
    const double *q = it->second.quaternion.coeffs().data();
    double cov_pp[9], cov_qq[9], cov_pq[9];
    covariance.GetCovarianceBlockInTangentSpace(p, p, cov_pp);
    covariance.GetCovarianceBlockInTangentSpace(q, q, cov_qq);
    covariance.GetCovarianceBlockInTangentSpace(p, q, cov_pq);

    it->second.covariance.block<3, 3>(0, 0) = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(cov_pp);
    it->second.covariance.block<3, 3>(2, 2) = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(cov_qq);
    it->second.covariance.block<3, 3>(0, 2) = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(cov_pq);
    it->second.covariance.block<3, 3>(2, 0) = it->second.covariance.block<3, 3>(0, 2).transpose();
  }

  return true;
}

bool CeresGraphOptimizer::EstimateCovarianceSE3(ceres::Problem *problem)
{
  ceres::Covariance::Options options;
  ceres::Covariance covariance(options);

  std::vector<std::pair<const double *, const double *>> covariance_blocks;
  for (MapVertexPose::const_iterator it = vertices_.begin(); it != vertices_.end(); ++it) {
    const double *p = it->second.xi.data();
    covariance_blocks.push_back(std::make_pair(p, p));
  }

  CHECK(covariance.Compute(covariance_blocks, problem));

  for (MapVertexPose::iterator it = vertices_.begin(); it != vertices_.end(); ++it) {
    const double *p = it->second.xi.data();
    double cov_pp[36];
    covariance.GetCovarianceBlockInTangentSpace(p, p, cov_pp);
    it->second.covariance = Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(cov_pp);
  }

  return true;
}

bool CeresGraphOptimizer::GetOptimizedPose(std::deque<Eigen::Isometry3f> &optimized_pose)
{
  optimized_pose.clear();
  for (MapVertexPose::iterator it = vertices_.begin(); it != vertices_.end(); ++it) {
    optimized_pose.push_back(GetOptimizedPose(it->first));
  }
  return true;
}
Eigen::Isometry3f CeresGraphOptimizer::GetOptimizedPose(int vertex_index)
{
  Eigen::Isometry3d pose;
  pose.setIdentity();
  pose.linear() = vertices_[vertex_index].quaternion.toRotationMatrix();
  pose.translation() = vertices_[vertex_index].position;

  Eigen::Isometry3f pose4f = pose.cast<float>();

  return pose4f;
}

Eigen::Isometry3f CeresGraphOptimizer::GetOptimizedPose()
{
  MapVertexPose::iterator it = vertices_.end();
  it--;
  return GetOptimizedPose(it->first);
}

Eigen::Matrix<float, 6, 6> CeresGraphOptimizer::GetOptimizedCovariance(int vertex_index)
{
  Eigen::Matrix<float, 6, 6> cov;
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++) cov(i, j) = float(vertices_[vertex_index].covariance(i, j));

  return cov;
}

bool CeresGraphOptimizer::UpdateGraphNode()
{
  return true;
}

int CeresGraphOptimizer::GetNodeNum()
{
  return vertices_.size();
}

void CeresGraphOptimizer::AddSe3Node(const Eigen::Isometry3f &pose, bool need_fix)
{
  unsigned int vertex_index = vertices_.size();
  AddSe3Node(pose, vertex_index, need_fix);
}
void CeresGraphOptimizer::AddSe3Node(const Eigen::Isometry3f &pose, unsigned int vertex_index, bool need_fix)
{
  vertices_.insert(
      std::pair<unsigned int, VertexPose>(vertex_index, VertexPose(vertex_index, pose.cast<double>(), need_fix)));
}
void CeresGraphOptimizer::RemoveSe3Node(unsigned int vertex_index)
{
  vertices_.erase(vertex_index);
}
void CeresGraphOptimizer::RemoveSe3Node()
{
  vertices_.erase(vertices_.begin());
}
void CeresGraphOptimizer::AddSe3Edge(int vertex_index1, int vertex_index2, const Eigen::Isometry3f &relative_pose,
                                     const Eigen::MatrixXf covariance)
{
  // INFO("odom edge\t", covariance.diagonal().transpose());
  Matrix6d info = CalculateInformationMatrix(covariance.cast<double>());
  edges_.push_back(EdgePose(vertex_index1, vertex_index2, relative_pose.cast<double>(), info));
}
void CeresGraphOptimizer::AddSe3PreintEdge(int vertex_index1, int vertex_index2, const Eigen::Isometry3f &relative_pose,
                                           const Eigen::MatrixXf covariance)
{
  // INFO("preint edge\t", covariance.diagonal().transpose());
  Matrix6d info = CalculateInformationMatrix(covariance.cast<double>());
  edges_preint_.push_back(EdgePose(vertex_index1, vertex_index2, relative_pose.cast<double>(), info));
}
void CeresGraphOptimizer::AddSe3LoopEdge(int vertex_index1, int vertex_index2, const Eigen::Isometry3f &relative_pose,
                                         const Eigen::MatrixXf covariance)
{
  Matrix6d info = CalculateInformationMatrix(covariance.cast<double>());
  edges_loop_.push_back(EdgePose(vertex_index1, vertex_index2, relative_pose.cast<double>(), info));
}
void CeresGraphOptimizer::AddRobustKernel(std::string kernel_type, double kernel_size)
{
  if (kernel_type == "NONE") {
    loss_function_ = nullptr;
  } else if (kernel_type == "Huber") {
    loss_function_ = new ceres::HuberLoss(kernel_size);
  } else if (kernel_type == "Cauchy") {
    loss_function_ = new ceres::CauchyLoss(kernel_size);
  }
}
Eigen::MatrixXd CeresGraphOptimizer::CalculateDiagMatrix(Eigen::VectorXd noise)
{
  Eigen::MatrixXd information_matrix = Eigen::MatrixXd::Identity(noise.rows(), noise.rows());
  for (int i = 0; i < noise.rows(); i++) {
    information_matrix(i, i) /= (noise(i) * noise(i));
  }
  return information_matrix;
}

void CeresGraphOptimizer::AddSe3PriorXYZEdge(int se3_vertex_index, const Eigen::Vector3f &xyz, Eigen::VectorXf noise)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = xyz.cast<double>();
  Matrix6d info = Matrix6d::Zero();
  info.block<3, 3>(0, 0) = CalculateDiagMatrix(noise.cast<double>());
  edges_gnss_.push_back(EdgePose(se3_vertex_index, se3_vertex_index, pose, info));
}
void CeresGraphOptimizer::AddSe3PriorZEdge(int vertex_index1, const Eigen::Vector2f height_range,
                                           const Eigen::VectorXf noise)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation()(2) = height_range(0);
  Matrix6d info = Matrix6d::Zero();
  info.block<1, 1>(2, 2) = CalculateDiagMatrix(noise.cast<double>());
  edges_abs_height_.push_back(EdgePose(vertex_index1, vertex_index1, pose, info));
}
void CeresGraphOptimizer::AddSe3PriordZEdge(int vertex_index1, int vertex_index2, const double delta_z,
                                            const Eigen::VectorXf noise)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation()(2) = delta_z;
  Matrix6d info = Matrix6d::Zero();
  info.block<1, 1>(2, 2) = CalculateDiagMatrix(noise.cast<double>());
  // INFO("relative height edge:\t", info(2, 2));
  edges_rel_height_.push_back(EdgePose(vertex_index1, vertex_index2, pose, info));
}
void CeresGraphOptimizer::AddSe3PriordRPEdge(int vertex_index1, int vertex_index2, const Eigen::Vector2f delta_RP,
                                             const Eigen::VectorXf noise)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation().block<2, 1>(0, 0) = delta_RP.cast<double>();
  Matrix6d info = Matrix6d::Zero();
  info.block<2, 2>(0, 0) = CalculateDiagMatrix(noise.cast<double>());
  // INFO("roll pitch edge:\t", info.block<2, 2>(0, 0).diagonal().transpose());
  edges_roll_pitch_.push_back(EdgePose(vertex_index1, vertex_index2, pose, info));
}
Eigen::MatrixXd CeresGraphOptimizer::CalculateInformationMatrix(Eigen::MatrixXd cov)
{
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
  es.compute(cov);
  Eigen::VectorXd Lambda = es.eigenvalues();
  Lambda = Lambda.cwiseAbs();
  Eigen::MatrixXd U = es.eigenvectors();
  Eigen::MatrixXd inv_Lambda = Eigen::MatrixXd::Zero(cov.rows(), cov.rows());
  for (int i = 0; i < inv_Lambda.rows(); i++) {
    if (Lambda(i) > 1e-7) {
      inv_Lambda(i, i) = 1.0 / Lambda(i);
    }
  }
  Eigen::MatrixXd info = U * inv_Lambda * U.transpose();
  return info;
}
void CeresGraphOptimizer::DisplayInformation()
{
  if (edges_.size() != 0) REMIND("odom edge:\ntrace(info) = " + std::to_string(EvaluateEdgeInformation(edges_, 1.0)));
  if (edges_preint_.size() != 0)
    REMIND("preint edge:\nweight = " + std::to_string(weight_preint_edge_) +
           "\ntrace(info) = " + std::to_string(EvaluateEdgeInformation(edges_preint_, weight_preint_edge_)));
  if (edges_loop_.size() != 0)
    REMIND("loop edge:\nweight = " + std::to_string(weight_loop_edge_) +
           "\ntrace(info) = " + std::to_string(EvaluateEdgeInformation(edges_loop_, weight_loop_edge_)));
  if (edges_gnss_.size() != 0)
    REMIND("gnss edge:\nweight = " + std::to_string(weight_gnss_edge_) +
           "\ntrace(info) = " + std::to_string(EvaluateEdgeInformation(edges_gnss_, weight_gnss_edge_)));
  std::cout << std::endl;
}
double CeresGraphOptimizer::EvaluateEdgeInformation(const DequeEdgePose &edges, double weight)
{
  double result = 0.0;
  for (DequeEdgePose::const_iterator it = edges.begin(); it != edges.end(); ++it) {
    result += it->information.trace() * weight * weight;
  }
  result /= edges.size();
  return result;
}
}  // namespace vision_localization
