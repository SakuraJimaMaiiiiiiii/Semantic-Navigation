#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace native_ego {

using Vec3 = std::array<double, 3>;
using Index3 = std::array<int, 3>;

struct PlannerConfig {
  double control_point_spacing{0.35};
  double clearance{0.45};
  double knot_interval{0.35};
  double max_velocity{0.8};
  double max_acceleration{1.2};
  double feasibility_tolerance{0.05};
  int optimization_iterations{200};
  int lbfgs_memory{16};
  int rebound_max_restarts{6};
  double smoothness_weight{1.0};
  double collision_weight{5.0};
  double feasibility_weight{0.2};
  double fitness_weight{0.15};
  double collision_check_step{0.08};
  int max_search_nodes{60000};
};

struct GridView {
  const std::uint8_t* blocked{nullptr};
  Index3 shape{0, 0, 0};
  Vec3 origin{0.0, 0.0, 0.0};
  double resolution{0.0};

  bool inside(const Index3& index) const;
  std::size_t flatIndex(const Index3& index) const;
  bool isBlocked(const Index3& index) const;
  Index3 worldToGrid(const Vec3& point) const;
  Vec3 gridToWorld(const Index3& index) const;
};

struct TrajectoryResult {
  bool success{false};
  std::string status{"optimization_failed"};
  std::vector<Vec3> control_points;
  double knot_interval{0.0};
  int rebound_count{0};
  bool time_reallocated{false};
};

/** ROS-free adapter of the original EGO-Planner planning pipeline. */
class EgoPlanner {
 public:
  explicit EgoPlanner(PlannerConfig config = {});

  std::vector<Index3> astarSearch(
      const GridView& grid,
      const Index3& start,
      const Index3& goal) const;

  TrajectoryResult reboundReplan(
      const GridView& grid,
      const std::vector<Vec3>& seed_path,
      const Vec3& start_velocity = {0.0, 0.0, 0.0},
      const Vec3& start_acceleration = {0.0, 0.0, 0.0},
      const Vec3& target_velocity = {0.0, 0.0, 0.0}) const;

  static Vec3 evaluateBspline(
      const std::vector<Vec3>& control_points,
      double knot_interval,
      double trajectory_time);
  static Vec3 evaluateVelocity(
      const std::vector<Vec3>& control_points,
      double knot_interval,
      double trajectory_time);
  static Vec3 evaluateAcceleration(
      const std::vector<Vec3>& control_points,
      double knot_interval,
      double trajectory_time);

 private:
  struct ReboundConstraint {
    Vec3 base_point{0.0, 0.0, 0.0};
    Vec3 direction{0.0, 0.0, 0.0};
  };
  using ConstraintList = std::vector<std::vector<ReboundConstraint>>;

  PlannerConfig config_;

  std::vector<Vec3> buildInitialPointSet(
      const std::vector<Vec3>& seed_path,
      const Vec3& start_velocity,
      const Vec3& start_acceleration,
      const Vec3& target_velocity,
      double& knot_interval) const;
  std::vector<Vec3> sampleBoundaryPolynomial(
      const Vec3& start,
      const Vec3& start_velocity,
      const Vec3& start_acceleration,
      const Vec3& target,
      const Vec3& target_velocity,
      double& knot_interval) const;
  std::vector<Vec3> resamplePolyline(
      const std::vector<Vec3>& path,
      double spacing) const;
  std::vector<Vec3> parameterizeToBspline(
      const std::vector<Vec3>& point_set,
      double knot_interval,
      const Vec3& start_velocity,
      const Vec3& target_velocity,
      const Vec3& start_acceleration,
      const Vec3& target_acceleration) const;

  bool initializeReboundConstraints(
      const GridView& grid,
      const std::vector<Vec3>& points,
      ConstraintList& constraints) const;
  bool optimizeRebound(
      const GridView& grid,
      std::vector<Vec3>& points,
      ConstraintList& constraints,
      double knot_interval,
      int& rebound_count) const;
  bool optimizeRefine(
      std::vector<Vec3>& points,
      const std::vector<Vec3>& reference_positions,
      double knot_interval) const;

  double feasibleRatio(
      const std::vector<Vec3>& points,
      double knot_interval) const;
  bool trajectoryIsFree(
      const GridView& grid,
      const std::vector<Vec3>& control_points,
      double knot_interval) const;
  bool transitionIsFree(
      const GridView& grid,
      const Index3& start,
      const Index3& end) const;
};

}  // namespace native_ego
