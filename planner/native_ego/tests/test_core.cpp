#include "native_ego/ego_planner.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

int main() {
  using native_ego::EgoPlanner;
  using native_ego::GridView;
  using native_ego::Index3;
  using native_ego::Vec3;

  constexpr int side = 7;
  std::vector<std::uint8_t> blocked(side * side * side, 0);
  GridView grid{
      blocked.data(), {side, side, side},
      {0.0, 0.0, 0.0}, 0.2};
  EgoPlanner planner;

  const auto path = planner.astarSearch(
      grid, Index3{1, 1, 1}, Index3{5, 5, 5});
  assert(!path.empty());
  assert(path.front() == Index3({1, 1, 1}));
  assert(path.back() == Index3({5, 5, 5}));

  const Vec3 start{0.0, 0.0, 0.0};
  const Vec3 end{2.0, 1.0, -0.5};
  const std::vector<Vec3> points{start, start, start, end, end, end};
  const Vec3 first = EgoPlanner::evaluateBspline(points, 0.2, 0.0);
  const Vec3 last = EgoPlanner::evaluateBspline(points, 0.2, 0.6);
  for (int axis = 0; axis < 3; ++axis) {
    assert(std::abs(first[axis] - start[axis]) < 1e-9);
    assert(std::abs(last[axis] - end[axis]) < 1e-9);
  }

  const Vec3 requested_velocity{0.3, -0.1, 0.0};
  const auto trajectory = planner.reboundReplan(
      grid,
      std::vector<Vec3>{{0.3, 0.3, 0.3}, {1.0, 0.6, 0.4}},
      requested_velocity,
      Vec3{0.0, 0.0, 0.0},
      Vec3{0.0, 0.0, 0.0});
  assert(trajectory.success);
  const Vec3 actual_velocity = EgoPlanner::evaluateVelocity(
      trajectory.control_points, trajectory.knot_interval, 0.0);
  for (int axis = 0; axis < 3; ++axis) {
    assert(std::abs(actual_velocity[axis] - requested_velocity[axis]) < 0.03);
  }

  constexpr int obstacle_side = 21;
  std::vector<std::uint8_t> obstacle_blocked(
      obstacle_side * obstacle_side * obstacle_side, 0);
  const auto flat = [=](int x, int y, int z) {
    return (x * obstacle_side + y) * obstacle_side + z;
  };
  for (int x = 8; x <= 12; ++x) {
    for (int y = 8; y <= 12; ++y) {
      for (int z = 8; z <= 12; ++z) {
        obstacle_blocked[flat(x, y, z)] = 1;
      }
    }
  }
  GridView obstacle_grid{
      obstacle_blocked.data(),
      {obstacle_side, obstacle_side, obstacle_side},
      {0.0, 0.0, 0.0}, 0.2};
  const auto rebound = planner.reboundReplan(
      obstacle_grid,
      std::vector<Vec3>{{0.5, 2.1, 2.1}, {3.5, 2.1, 2.1}});
  assert(rebound.success);
  assert(rebound.rebound_count > 0);
  const double duration =
      (rebound.control_points.size() - 3.0) * rebound.knot_interval;
  for (double time = 0.0; time <= duration; time += 0.02) {
    const Vec3 point = EgoPlanner::evaluateBspline(
        rebound.control_points, rebound.knot_interval, time);
    assert(!obstacle_grid.isBlocked(obstacle_grid.worldToGrid(point)));
  }
  return 0;
}
