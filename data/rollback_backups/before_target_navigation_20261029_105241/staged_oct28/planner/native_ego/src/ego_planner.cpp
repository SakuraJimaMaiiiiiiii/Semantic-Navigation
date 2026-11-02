#include "native_ego/ego_planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>

namespace native_ego {
namespace {

constexpr int kOrder = 3;
constexpr double kEpsilon = 1e-9;

Vec3 add(const Vec3& a, const Vec3& b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 subtract(const Vec3& a, const Vec3& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 multiply(const Vec3& value, double scale) {
  return {value[0] * scale, value[1] * scale, value[2] * scale};
}

double dot(const Vec3& a, const Vec3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double squaredNorm(const Vec3& value) { return dot(value, value); }
double norm(const Vec3& value) { return std::sqrt(squaredNorm(value)); }

Vec3 normalized(const Vec3& value) {
  const double length = norm(value);
  return length <= kEpsilon ? Vec3{0.0, 0.0, 0.0}
                            : multiply(value, 1.0 / length);
}

double indexDistance(const Index3& a, const Index3& b) {
  return norm({static_cast<double>(a[0] - b[0]),
               static_cast<double>(a[1] - b[1]),
               static_cast<double>(a[2] - b[2])});
}

struct SearchNode {
  double estimated_total;
  double cost;
  std::size_t flat_index;
  Index3 index;
};

struct NodeGreater {
  bool operator()(const SearchNode& a, const SearchNode& b) const {
    return a.estimated_total > b.estimated_total;
  }
};

using Vector = std::vector<double>;
using Objective = std::function<double(const Vector&, Vector&)>;

double vectorDot(const Vector& a, const Vector& b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double vectorNorm(const Vector& value) {
  return std::sqrt(vectorDot(value, value));
}

bool limitedMemoryBfgs(
    Vector& x,
    const Objective& objective,
    int max_iterations,
    int memory,
    double gradient_tolerance) {
  Vector gradient(x.size());
  double cost = objective(x, gradient);
  if (!std::isfinite(cost)) {
    return false;
  }
  std::vector<Vector> s_history;
  std::vector<Vector> y_history;
  std::vector<double> rho_history;

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    if (vectorNorm(gradient) /
            std::sqrt(std::max<std::size_t>(1, gradient.size())) <
        gradient_tolerance) {
      return true;
    }

    Vector direction = gradient;
    std::vector<double> alpha(s_history.size(), 0.0);
    for (std::size_t reverse = s_history.size(); reverse-- > 0;) {
      alpha[reverse] =
          rho_history[reverse] * vectorDot(s_history[reverse], direction);
      for (std::size_t i = 0; i < direction.size(); ++i) {
        direction[i] -= alpha[reverse] * y_history[reverse][i];
      }
    }
    if (!s_history.empty()) {
      const Vector& last_s = s_history.back();
      const Vector& last_y = y_history.back();
      const double yy = vectorDot(last_y, last_y);
      const double scale = yy > kEpsilon ? vectorDot(last_s, last_y) / yy : 1.0;
      for (double& value : direction) {
        value *= std::clamp(scale, 1e-4, 1e4);
      }
    }
    for (std::size_t h = 0; h < s_history.size(); ++h) {
      const double beta = rho_history[h] * vectorDot(y_history[h], direction);
      for (std::size_t i = 0; i < direction.size(); ++i) {
        direction[i] += s_history[h][i] * (alpha[h] - beta);
      }
    }
    for (double& value : direction) {
      value = -value;
    }
    double directional_derivative = vectorDot(gradient, direction);
    if (!std::isfinite(directional_derivative) ||
        directional_derivative >= -kEpsilon) {
      for (std::size_t i = 0; i < direction.size(); ++i) {
        direction[i] = -gradient[i];
      }
      directional_derivative = -vectorDot(gradient, gradient);
    }

    double step = 1.0;
    Vector candidate(x.size());
    Vector candidate_gradient(x.size());
    double candidate_cost = std::numeric_limits<double>::infinity();
    bool accepted = false;
    for (int line_search = 0; line_search < 24; ++line_search) {
      for (std::size_t i = 0; i < x.size(); ++i) {
        candidate[i] = x[i] + step * direction[i];
      }
      candidate_cost = objective(candidate, candidate_gradient);
      if (std::isfinite(candidate_cost) &&
          candidate_cost <= cost + 1e-4 * step * directional_derivative) {
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if (!accepted) {
      return vectorNorm(gradient) < 0.1;
    }

    Vector s(x.size());
    Vector y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      s[i] = candidate[i] - x[i];
      y[i] = candidate_gradient[i] - gradient[i];
    }
    const double ys = vectorDot(y, s);
    if (ys > 1e-10) {
      if (static_cast<int>(s_history.size()) == memory) {
        s_history.erase(s_history.begin());
        y_history.erase(y_history.begin());
        rho_history.erase(rho_history.begin());
      }
      s_history.push_back(std::move(s));
      y_history.push_back(std::move(y));
      rho_history.push_back(1.0 / ys);
    }
    x = std::move(candidate);
    gradient = std::move(candidate_gradient);
    cost = candidate_cost;
  }
  return true;
}

Vector solveLeastSquares(
    const std::vector<Vector>& matrix,
    const Vector& values) {
  if (matrix.empty() || matrix.size() != values.size()) {
    return {};
  }
  const std::size_t columns = matrix.front().size();
  std::vector<Vector> system(columns, Vector(columns + 1, 0.0));
  for (std::size_t row = 0; row < matrix.size(); ++row) {
    for (std::size_t i = 0; i < columns; ++i) {
      system[i][columns] += matrix[row][i] * values[row];
      for (std::size_t j = 0; j < columns; ++j) {
        system[i][j] += matrix[row][i] * matrix[row][j];
      }
    }
  }
  for (std::size_t i = 0; i < columns; ++i) {
    system[i][i] += 1e-10;
  }
  for (std::size_t pivot = 0; pivot < columns; ++pivot) {
    std::size_t best = pivot;
    for (std::size_t row = pivot + 1; row < columns; ++row) {
      if (std::abs(system[row][pivot]) > std::abs(system[best][pivot])) {
        best = row;
      }
    }
    if (std::abs(system[best][pivot]) < 1e-12) {
      return {};
    }
    std::swap(system[pivot], system[best]);
    const double divisor = system[pivot][pivot];
    for (std::size_t column = pivot; column <= columns; ++column) {
      system[pivot][column] /= divisor;
    }
    for (std::size_t row = 0; row < columns; ++row) {
      if (row == pivot) {
        continue;
      }
      const double factor = system[row][pivot];
      for (std::size_t column = pivot; column <= columns; ++column) {
        system[row][column] -= factor * system[pivot][column];
      }
    }
  }
  Vector solution(columns);
  for (std::size_t i = 0; i < columns; ++i) {
    solution[i] = system[i][columns];
  }
  return solution;
}

Vector flattenMovable(const std::vector<Vec3>& points) {
  Vector result;
  if (points.size() <= 2 * kOrder) {
    return result;
  }
  result.reserve((points.size() - 2 * kOrder) * 3);
  for (std::size_t i = kOrder; i + kOrder < points.size(); ++i) {
    result.insert(result.end(), points[i].begin(), points[i].end());
  }
  return result;
}

void assignMovable(std::vector<Vec3>& points, const Vector& values) {
  std::size_t offset = 0;
  for (std::size_t i = kOrder; i + kOrder < points.size(); ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      points[i][axis] = values[offset++];
    }
  }
}

Vector movableGradient(const std::vector<Vec3>& gradient) {
  return flattenMovable(gradient);
}

bool segmentBlocked(const GridView& grid, const Vec3& start, const Vec3& end) {
  const double distance = norm(subtract(end, start));
  const int count = std::max(
      1, static_cast<int>(std::ceil(distance / (0.5 * grid.resolution))));
  for (int sample = 0; sample <= count; ++sample) {
    const double ratio = static_cast<double>(sample) / count;
    const Vec3 point = add(start, multiply(subtract(end, start), ratio));
    if (grid.isBlocked(grid.worldToGrid(point))) {
      return true;
    }
  }
  return false;
}

}  // namespace

bool GridView::inside(const Index3& index) const {
  return index[0] >= 0 && index[1] >= 0 && index[2] >= 0 &&
         index[0] < shape[0] && index[1] < shape[1] && index[2] < shape[2];
}

std::size_t GridView::flatIndex(const Index3& index) const {
  return (static_cast<std::size_t>(index[0]) * shape[1] + index[1]) *
             shape[2] +
         index[2];
}

bool GridView::isBlocked(const Index3& index) const {
  return !inside(index) || blocked == nullptr || blocked[flatIndex(index)] != 0;
}

Index3 GridView::worldToGrid(const Vec3& point) const {
  return {static_cast<int>(std::floor((point[0] - origin[0]) / resolution)),
          static_cast<int>(std::floor((point[1] - origin[1]) / resolution)),
          static_cast<int>(std::floor((point[2] - origin[2]) / resolution))};
}

Vec3 GridView::gridToWorld(const Index3& index) const {
  return {origin[0] + (static_cast<double>(index[0]) + 0.5) * resolution,
          origin[1] + (static_cast<double>(index[1]) + 0.5) * resolution,
          origin[2] + (static_cast<double>(index[2]) + 0.5) * resolution};
}

EgoPlanner::EgoPlanner(PlannerConfig config) : config_(config) {
  if (config_.control_point_spacing <= 0.0 || config_.clearance <= 0.0 ||
      config_.knot_interval <= 0.0 || config_.max_velocity <= 0.0 ||
      config_.max_acceleration <= 0.0 || config_.feasibility_tolerance < 0.0 ||
      config_.optimization_iterations < 1 || config_.lbfgs_memory < 1 ||
      config_.rebound_max_restarts < 1 || config_.collision_check_step <= 0.0 ||
      config_.max_search_nodes < 1) {
    throw std::invalid_argument("Native EGO configuration is invalid.");
  }
}

bool EgoPlanner::transitionIsFree(
    const GridView& grid, const Index3& start, const Index3& end) const {
  Index3 delta{end[0] - start[0], end[1] - start[1], end[2] - start[2]};
  for (int axis = 0; axis < 3; ++axis) {
    if (std::abs(delta[axis]) > 1) {
      return false;
    }
  }
  // Deliberately match upstream dyn_a_star: occupancy is checked at the
  // neighbor voxel. The inflated map supplies the required corner clearance.
  return !grid.isBlocked(end);
}

std::vector<Index3> EgoPlanner::astarSearch(
    const GridView& grid, const Index3& start, const Index3& goal) const {
  if (!grid.inside(start) || !grid.inside(goal) || grid.isBlocked(goal)) {
    return {};
  }
  const std::size_t count = static_cast<std::size_t>(grid.shape[0]) *
                            grid.shape[1] * grid.shape[2];
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> scores(count, infinity);
  std::vector<std::int64_t> parents(count, -1);
  std::vector<std::uint8_t> closed(count, 0);
  std::priority_queue<SearchNode, std::vector<SearchNode>, NodeGreater> open;
  const std::size_t start_flat = grid.flatIndex(start);
  scores[start_flat] = 0.0;
  open.push({indexDistance(start, goal), 0.0, start_flat, start});

  int expanded = 0;
  while (!open.empty() && expanded < config_.max_search_nodes) {
    const SearchNode current = open.top();
    open.pop();
    if (closed[current.flat_index] != 0 ||
        std::abs(current.cost - scores[current.flat_index]) > 1e-12) {
      continue;
    }
    closed[current.flat_index] = 1;
    ++expanded;
    if (current.index == goal) {
      std::vector<Index3> path;
      std::int64_t flat = static_cast<std::int64_t>(current.flat_index);
      while (flat >= 0) {
        const int z = static_cast<int>(flat % grid.shape[2]);
        const std::int64_t xy = flat / grid.shape[2];
        const int y = static_cast<int>(xy % grid.shape[1]);
        const int x = static_cast<int>(xy / grid.shape[1]);
        path.push_back({x, y, z});
        flat = parents[static_cast<std::size_t>(flat)];
      }
      std::reverse(path.begin(), path.end());
      return path;
    }
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0) {
            continue;
          }
          const Index3 next{current.index[0] + dx, current.index[1] + dy,
                            current.index[2] + dz};
          if (grid.isBlocked(next)) {
            continue;
          }
          const std::size_t next_flat = grid.flatIndex(next);
          if (closed[next_flat] != 0) {
            continue;
          }
          const double edge =
              std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz));
          const double candidate = current.cost + edge;
          if (candidate >= scores[next_flat]) {
            continue;
          }
          scores[next_flat] = candidate;
          parents[next_flat] = static_cast<std::int64_t>(current.flat_index);
          open.push({candidate + indexDistance(next, goal), candidate, next_flat,
                     next});
        }
      }
    }
  }
  return {};
}

std::vector<Vec3> EgoPlanner::resamplePolyline(
    const std::vector<Vec3>& path, double spacing) const {
  if (path.size() < 2) {
    return path;
  }
  std::vector<double> cumulative(path.size(), 0.0);
  for (std::size_t i = 1; i < path.size(); ++i) {
    cumulative[i] = cumulative[i - 1] + norm(subtract(path[i], path[i - 1]));
  }
  const double total = cumulative.back();
  if (total <= kEpsilon) {
    return {path.front()};
  }
  const int point_count =
      std::max(7, static_cast<int>(std::ceil(total / spacing)) + 1);
  std::vector<Vec3> result;
  result.reserve(static_cast<std::size_t>(point_count));
  std::size_t segment = 0;
  for (int sample = 0; sample < point_count; ++sample) {
    const double distance = total * sample / (point_count - 1.0);
    while (segment + 2 < path.size() && distance > cumulative[segment + 1]) {
      ++segment;
    }
    const double length = cumulative[segment + 1] - cumulative[segment];
    const double ratio = length <= kEpsilon
                             ? 0.0
                             : (distance - cumulative[segment]) / length;
    result.push_back(add(path[segment],
                         multiply(subtract(path[segment + 1], path[segment]),
                                  ratio)));
  }
  return result;
}

std::vector<Vec3> EgoPlanner::sampleBoundaryPolynomial(
    const Vec3& start,
    const Vec3& start_velocity,
    const Vec3& start_acceleration,
    const Vec3& target,
    const Vec3& target_velocity,
    double& knot_interval) const {
  const double distance = norm(subtract(target, start));
  const double velocity_switch =
      config_.max_velocity * config_.max_velocity / config_.max_acceleration;
  const double duration = std::max(
      2.0 * knot_interval,
      velocity_switch > distance
          ? std::sqrt(std::max(distance, 1e-6) / config_.max_acceleration)
          : (distance - velocity_switch) / config_.max_velocity +
                2.0 * config_.max_velocity / config_.max_acceleration);

  std::array<std::array<double, 6>, 3> coefficients{};
  for (int axis = 0; axis < 3; ++axis) {
    const double t = duration;
    const double c0 = start[axis];
    const double c1 = start_velocity[axis];
    const double c2 = 0.5 * start_acceleration[axis];
    const double r1 = target[axis] - (c0 + c1 * t + c2 * t * t);
    const double r2 = target_velocity[axis] - (c1 + 2.0 * c2 * t);
    const double r3 = -2.0 * c2;
    coefficients[axis] = {
        c0,
        c1,
        c2,
        10.0 * r1 / std::pow(t, 3) - 4.0 * r2 / std::pow(t, 2) +
            0.5 * r3 / t,
        -15.0 * r1 / std::pow(t, 4) + 7.0 * r2 / std::pow(t, 3) -
            r3 / std::pow(t, 2),
        6.0 * r1 / std::pow(t, 5) - 3.0 * r2 / std::pow(t, 4) +
            0.5 * r3 / std::pow(t, 3)};
  }

  int point_count = std::max(7, static_cast<int>(std::ceil(duration / knot_interval)) + 1);
  auto evaluate = [&](double time) {
    Vec3 point{};
    for (int axis = 0; axis < 3; ++axis) {
      double power = 1.0;
      for (double coefficient : coefficients[axis]) {
        point[axis] += coefficient * power;
        power *= time;
      }
    }
    return point;
  };
  while (true) {
    bool too_far = false;
    Vec3 previous = evaluate(0.0);
    for (int i = 1; i < point_count; ++i) {
      const Vec3 point = evaluate(duration * i / (point_count - 1.0));
      if (norm(subtract(point, previous)) > 1.5 * config_.control_point_spacing) {
        too_far = true;
        break;
      }
      previous = point;
    }
    if (!too_far) {
      break;
    }
    point_count = point_count * 3 / 2 + 1;
  }
  knot_interval = duration / (point_count - 1.0);
  std::vector<Vec3> result;
  result.reserve(static_cast<std::size_t>(point_count));
  for (int i = 0; i < point_count; ++i) {
    result.push_back(evaluate(duration * i / (point_count - 1.0)));
  }
  return result;
}

std::vector<Vec3> EgoPlanner::buildInitialPointSet(
    const std::vector<Vec3>& seed_path,
    const Vec3& start_velocity,
    const Vec3& start_acceleration,
    const Vec3& target_velocity,
    double& knot_interval) const {
  if (seed_path.size() < 2) {
    return {};
  }
  knot_interval = std::max(
      config_.knot_interval,
      1.2 * config_.control_point_spacing / config_.max_velocity);
  if (seed_path.size() == 2) {
    return sampleBoundaryPolynomial(seed_path.front(), start_velocity,
                                    start_acceleration, seed_path.back(),
                                    target_velocity, knot_interval);
  }
  const std::vector<Vec3> sampled =
      resamplePolyline(seed_path, config_.control_point_spacing);
  if (sampled.size() >= 2) {
    const double average = norm(subtract(sampled.back(), sampled.front())) /
                           std::max<std::size_t>(1, sampled.size() - 1);
    knot_interval = std::max(config_.knot_interval,
                             1.2 * average / config_.max_velocity);
  }
  return sampled;
}

std::vector<Vec3> EgoPlanner::parameterizeToBspline(
    const std::vector<Vec3>& point_set,
    double knot_interval,
    const Vec3& start_velocity,
    const Vec3& target_velocity,
    const Vec3& start_acceleration,
    const Vec3& target_acceleration) const {
  if (point_set.size() <= 3 || knot_interval <= 0.0) {
    return {};
  }
  const std::size_t k = point_set.size();
  const std::size_t rows = k + 4;
  const std::size_t columns = k + 2;
  std::vector<Vector> matrix(rows, Vector(columns, 0.0));
  for (std::size_t i = 0; i < k; ++i) {
    matrix[i][i] = 1.0 / 6.0;
    matrix[i][i + 1] = 4.0 / 6.0;
    matrix[i][i + 2] = 1.0 / 6.0;
  }
  const double velocity_scale = 0.5 / knot_interval;
  matrix[k][0] = -velocity_scale;
  matrix[k][2] = velocity_scale;
  matrix[k + 1][k - 1] = -velocity_scale;
  matrix[k + 1][k + 1] = velocity_scale;
  const double acceleration_scale = 1.0 / (knot_interval * knot_interval);
  matrix[k + 2][0] = acceleration_scale;
  matrix[k + 2][1] = -2.0 * acceleration_scale;
  matrix[k + 2][2] = acceleration_scale;
  matrix[k + 3][k - 1] = acceleration_scale;
  matrix[k + 3][k] = -2.0 * acceleration_scale;
  matrix[k + 3][k + 1] = acceleration_scale;

  std::vector<Vec3> points(columns);
  for (int axis = 0; axis < 3; ++axis) {
    Vector values(rows);
    for (std::size_t i = 0; i < k; ++i) {
      values[i] = point_set[i][axis];
    }
    values[k] = start_velocity[axis];
    values[k + 1] = target_velocity[axis];
    values[k + 2] = start_acceleration[axis];
    values[k + 3] = target_acceleration[axis];
    const Vector solution = solveLeastSquares(matrix, values);
    if (solution.size() != columns) {
      return {};
    }
    for (std::size_t i = 0; i < columns; ++i) {
      points[i][axis] = solution[i];
    }
  }
  return points;
}

bool EgoPlanner::initializeReboundConstraints(
    const GridView& grid,
    const std::vector<Vec3>& points,
    ConstraintList& constraints) const {
  constraints.assign(points.size(), {});
  if (points.size() <= 2 * kOrder) {
    return false;
  }
  std::vector<std::uint8_t> collision(points.size(), 0);
  for (std::size_t i = kOrder; i + kOrder < points.size(); ++i) {
    if (grid.isBlocked(grid.worldToGrid(points[i])) ||
        segmentBlocked(grid, points[i - 1], points[i]) ||
        segmentBlocked(grid, points[i], points[i + 1])) {
      collision[i] = 1;
    }
  }

  std::size_t cursor = kOrder;
  while (cursor + kOrder < points.size()) {
    if (collision[cursor] == 0) {
      ++cursor;
      continue;
    }
    const std::size_t first = cursor;
    while (cursor + kOrder < points.size() && collision[cursor] != 0) {
      ++cursor;
    }
    const std::size_t last = cursor - 1;
    const std::size_t in_index = std::max<std::size_t>(kOrder - 1, first - 1);
    const std::size_t out_index = std::min(points.size() - kOrder, last + 1);
    const Index3 start = grid.worldToGrid(points[in_index]);
    const Index3 goal = grid.worldToGrid(points[out_index]);
    const std::vector<Index3> cell_path = astarSearch(grid, start, goal);
    if (cell_path.size() < 2) {
      return false;
    }
    std::vector<Vec3> path;
    path.reserve(cell_path.size());
    for (const Index3& cell : cell_path) {
      path.push_back(grid.gridToWorld(cell));
    }
    path.front() = points[in_index];
    path.back() = points[out_index];

    Vec3 propagated_direction{0.0, 0.0, 0.0};
    Vec3 propagated_base{0.0, 0.0, 0.0};
    bool have_propagated = false;
    for (std::size_t point_index = first; point_index <= last; ++point_index) {
      const Vec3 tangent = subtract(
          points[std::min(point_index + 1, points.size() - 1)],
          points[point_index - 1]);
      Vec3 intersection{};
      double farthest = -1.0;
      for (std::size_t path_index = 0; path_index + 1 < path.size(); ++path_index) {
        const double first_value =
            dot(subtract(path[path_index], points[point_index]), tangent);
        const double second_value =
            dot(subtract(path[path_index + 1], points[point_index]), tangent);
        if (first_value * second_value > 0.0) {
          continue;
        }
        const double denominator = first_value - second_value;
        const double ratio = std::abs(denominator) <= kEpsilon
                                 ? 0.5
                                 : std::clamp(first_value / denominator, 0.0, 1.0);
        const Vec3 candidate = add(
            path[path_index],
            multiply(subtract(path[path_index + 1], path[path_index]), ratio));
        const double distance = norm(subtract(candidate, points[point_index]));
        if (distance > farthest) {
          farthest = distance;
          intersection = candidate;
        }
      }
      if (farthest <= 1e-5) {
        for (const Vec3& candidate : path) {
          const double distance = norm(subtract(candidate, points[point_index]));
          if (distance > farthest) {
            farthest = distance;
            intersection = candidate;
          }
        }
      }
      const Vec3 direction = normalized(subtract(intersection, points[point_index]));
      if (norm(direction) <= kEpsilon) {
        if (have_propagated) {
          constraints[point_index].push_back(
              {propagated_base, propagated_direction});
        }
        continue;
      }

      const double length = norm(subtract(intersection, points[point_index]));
      const int samples =
          std::max(1, static_cast<int>(std::ceil(length / grid.resolution)));
      Vec3 base = intersection;
      Vec3 previous = intersection;
      for (int sample = 1; sample <= samples; ++sample) {
        const double ratio = static_cast<double>(sample) / samples;
        const Vec3 candidate = add(
            intersection,
            multiply(subtract(points[point_index], intersection), ratio));
        if (grid.isBlocked(grid.worldToGrid(candidate))) {
          base = previous;
          break;
        }
        previous = candidate;
        base = candidate;
      }
      constraints[point_index].push_back({base, direction});
      propagated_base = base;
      propagated_direction = direction;
      have_propagated = true;
    }
  }
  return true;
}

bool EgoPlanner::optimizeRebound(
    const GridView& grid,
    std::vector<Vec3>& points,
    ConstraintList& constraints,
    double knot_interval,
    int& rebound_count) const {
  double collision_weight = config_.collision_weight;
  for (int restart = 0; restart < config_.rebound_max_restarts; ++restart) {
    const std::vector<Vec3> fixed_template = points;
    Vector variables = flattenMovable(points);
    if (variables.empty()) {
      return false;
    }
    const Objective objective = [&](const Vector& values, Vector& output) {
      std::vector<Vec3> q = fixed_template;
      assignMovable(q, values);
      std::vector<Vec3> gradient(q.size(), {0.0, 0.0, 0.0});
      double smoothness_cost = 0.0;
      for (std::size_t i = 0; i + 3 < q.size(); ++i) {
        const Vec3 jerk = add(
            subtract(q[i + 3], multiply(q[i + 2], 3.0)),
            subtract(multiply(q[i + 1], 3.0), q[i]));
        smoothness_cost += squaredNorm(jerk);
        const Vec3 value = multiply(jerk, 2.0 * config_.smoothness_weight);
        gradient[i] = subtract(gradient[i], value);
        gradient[i + 1] = add(gradient[i + 1], multiply(value, 3.0));
        gradient[i + 2] = subtract(gradient[i + 2], multiply(value, 3.0));
        gradient[i + 3] = add(gradient[i + 3], value);
      }

      double distance_cost = 0.0;
      const double demarcation = config_.clearance;
      const double a = 3.0 * demarcation;
      const double b = -3.0 * demarcation * demarcation;
      const double c = demarcation * demarcation * demarcation;
      for (std::size_t i = kOrder; i + kOrder < q.size(); ++i) {
        for (const ReboundConstraint& constraint : constraints[i]) {
          const double distance =
              dot(subtract(q[i], constraint.base_point), constraint.direction);
          const double error = config_.clearance - distance;
          if (error <= 0.0) {
            continue;
          }
          double derivative = 0.0;
          if (error < demarcation) {
            distance_cost += error * error * error;
            derivative = -3.0 * error * error;
          } else {
            distance_cost += a * error * error + b * error + c;
            derivative = -(2.0 * a * error + b);
          }
          gradient[i] = add(
              gradient[i],
              multiply(constraint.direction, collision_weight * derivative));
        }
      }

      double feasibility_cost = 0.0;
      const double inverse_dt2 = 1.0 / (knot_interval * knot_interval);
      for (std::size_t i = 0; i + 1 < q.size(); ++i) {
        for (int axis = 0; axis < 3; ++axis) {
          const double velocity =
              (q[i + 1][axis] - q[i][axis]) / knot_interval;
          double difference = 0.0;
          if (velocity > config_.max_velocity) {
            difference = velocity - config_.max_velocity;
          } else if (velocity < -config_.max_velocity) {
            difference = velocity + config_.max_velocity;
          }
          if (difference == 0.0) {
            continue;
          }
          feasibility_cost += difference * difference * inverse_dt2;
          const double derivative = 2.0 * difference / knot_interval * inverse_dt2;
          gradient[i][axis] -= config_.feasibility_weight * derivative;
          gradient[i + 1][axis] += config_.feasibility_weight * derivative;
        }
      }
      for (std::size_t i = 0; i + 2 < q.size(); ++i) {
        for (int axis = 0; axis < 3; ++axis) {
          const double acceleration =
              (q[i + 2][axis] - 2.0 * q[i + 1][axis] + q[i][axis]) *
              inverse_dt2;
          double difference = 0.0;
          if (acceleration > config_.max_acceleration) {
            difference = acceleration - config_.max_acceleration;
          } else if (acceleration < -config_.max_acceleration) {
            difference = acceleration + config_.max_acceleration;
          }
          if (difference == 0.0) {
            continue;
          }
          feasibility_cost += difference * difference;
          const double derivative = 2.0 * difference * inverse_dt2;
          gradient[i][axis] += config_.feasibility_weight * derivative;
          gradient[i + 1][axis] -=
              2.0 * config_.feasibility_weight * derivative;
          gradient[i + 2][axis] += config_.feasibility_weight * derivative;
        }
      }
      output = movableGradient(gradient);
      return config_.smoothness_weight * smoothness_cost +
             collision_weight * distance_cost +
             config_.feasibility_weight * feasibility_cost;
    };
    limitedMemoryBfgs(variables, objective, config_.optimization_iterations,
                      config_.lbfgs_memory, 0.01);
    assignMovable(points, variables);
    if (trajectoryIsFree(grid, points, knot_interval)) {
      return true;
    }
    ++rebound_count;
    if (!initializeReboundConstraints(grid, points, constraints)) {
      return false;
    }
    collision_weight *= 2.0;
  }
  return false;
}

bool EgoPlanner::optimizeRefine(
    std::vector<Vec3>& points,
    const std::vector<Vec3>& reference_positions,
    double knot_interval) const {
  const std::vector<Vec3> fixed_template = points;
  Vector variables = flattenMovable(points);
  if (variables.empty()) {
    return false;
  }
  const Objective objective = [&](const Vector& values, Vector& output) {
    std::vector<Vec3> q = fixed_template;
    assignMovable(q, values);
    std::vector<Vec3> gradient(q.size(), {0.0, 0.0, 0.0});
    double smoothness_cost = 0.0;
    for (std::size_t i = 0; i + 3 < q.size(); ++i) {
      const Vec3 jerk = add(subtract(q[i + 3], multiply(q[i + 2], 3.0)),
                            subtract(multiply(q[i + 1], 3.0), q[i]));
      smoothness_cost += squaredNorm(jerk);
      const Vec3 value = multiply(jerk, 2.0 * config_.smoothness_weight);
      gradient[i] = subtract(gradient[i], value);
      gradient[i + 1] = add(gradient[i + 1], multiply(value, 3.0));
      gradient[i + 2] = subtract(gradient[i + 2], multiply(value, 3.0));
      gradient[i + 3] = add(gradient[i + 3], value);
    }

    double fitness_cost = 0.0;
    for (std::size_t i = kOrder - 1; i + kOrder - 1 < q.size(); ++i) {
      const Vec3 position = multiply(
          add(add(q[i - 1], multiply(q[i], 4.0)), q[i + 1]), 1.0 / 6.0);
      const Vec3 error = subtract(position, reference_positions[i]);
      const Vec3 tangent = normalized(subtract(
          reference_positions[std::min(i + 1, reference_positions.size() - 1)],
          reference_positions[i - 1]));
      const double along = dot(error, tangent);
      fitness_cost += squaredNorm(error) - 0.96 * along * along;
      const Vec3 derivative =
          multiply(subtract(error, multiply(tangent, 0.96 * along)), 2.0);
      gradient[i - 1] = add(
          gradient[i - 1], multiply(derivative, config_.fitness_weight / 6.0));
      gradient[i] = add(
          gradient[i], multiply(derivative, 4.0 * config_.fitness_weight / 6.0));
      gradient[i + 1] = add(
          gradient[i + 1], multiply(derivative, config_.fitness_weight / 6.0));
    }

    double feasibility_cost = 0.0;
    const double inverse_dt2 = 1.0 / (knot_interval * knot_interval);
    for (std::size_t i = 0; i + 1 < q.size(); ++i) {
      for (int axis = 0; axis < 3; ++axis) {
        const double velocity = (q[i + 1][axis] - q[i][axis]) / knot_interval;
        const double difference =
            velocity > config_.max_velocity
                ? velocity - config_.max_velocity
                : (velocity < -config_.max_velocity
                       ? velocity + config_.max_velocity
                       : 0.0);
        if (difference == 0.0) {
          continue;
        }
        feasibility_cost += difference * difference * inverse_dt2;
        const double derivative = 2.0 * difference / knot_interval * inverse_dt2;
        gradient[i][axis] -= config_.feasibility_weight * derivative;
        gradient[i + 1][axis] += config_.feasibility_weight * derivative;
      }
    }
    for (std::size_t i = 0; i + 2 < q.size(); ++i) {
      for (int axis = 0; axis < 3; ++axis) {
        const double acceleration =
            (q[i + 2][axis] - 2.0 * q[i + 1][axis] + q[i][axis]) *
            inverse_dt2;
        const double difference =
            acceleration > config_.max_acceleration
                ? acceleration - config_.max_acceleration
                : (acceleration < -config_.max_acceleration
                       ? acceleration + config_.max_acceleration
                       : 0.0);
        if (difference == 0.0) {
          continue;
        }
        feasibility_cost += difference * difference;
        const double derivative = 2.0 * difference * inverse_dt2;
        gradient[i][axis] += config_.feasibility_weight * derivative;
        gradient[i + 1][axis] -=
            2.0 * config_.feasibility_weight * derivative;
        gradient[i + 2][axis] += config_.feasibility_weight * derivative;
      }
    }
    output = movableGradient(gradient);
    return config_.smoothness_weight * smoothness_cost +
           config_.fitness_weight * fitness_cost +
           config_.feasibility_weight * feasibility_cost;
  };
  const bool success = limitedMemoryBfgs(
      variables, objective, config_.optimization_iterations,
      config_.lbfgs_memory, 0.001);
  assignMovable(points, variables);
  return success;
}

double EgoPlanner::feasibleRatio(
    const std::vector<Vec3>& points, double knot_interval) const {
  double max_velocity = 0.0;
  double max_acceleration = 0.0;
  for (std::size_t i = 0; i + 1 < points.size(); ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      max_velocity = std::max(
          max_velocity,
          std::abs(points[i + 1][axis] - points[i][axis]) / knot_interval);
    }
  }
  for (std::size_t i = 0; i + 2 < points.size(); ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      max_acceleration = std::max(
          max_acceleration,
          std::abs(points[i + 2][axis] - 2.0 * points[i + 1][axis] +
                   points[i][axis]) /
              (knot_interval * knot_interval));
    }
  }
  return std::max(max_velocity / config_.max_velocity,
                  std::sqrt(max_acceleration / config_.max_acceleration));
}

Vec3 EgoPlanner::evaluateBspline(
    const std::vector<Vec3>& points,
    double knot_interval,
    double trajectory_time) {
  if (points.size() < 4 || knot_interval <= 0.0) {
    throw std::invalid_argument("Invalid B-spline trajectory.");
  }
  const double duration = (points.size() - 3.0) * knot_interval;
  const double time = std::clamp(trajectory_time, 0.0, duration);
  const double scaled = time / knot_interval;
  const std::size_t segment = std::min(
      static_cast<std::size_t>(std::floor(scaled)), points.size() - 4);
  const double u = time >= duration ? 1.0 : scaled - segment;
  const double u2 = u * u;
  const double u3 = u2 * u;
  const std::array<double, 4> basis{
      std::pow(1.0 - u, 3) / 6.0,
      (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0,
      (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0,
      u3 / 6.0};
  Vec3 result{};
  for (std::size_t i = 0; i < 4; ++i) {
    result = add(result, multiply(points[segment + i], basis[i]));
  }
  return result;
}

Vec3 EgoPlanner::evaluateVelocity(
    const std::vector<Vec3>& points,
    double knot_interval,
    double trajectory_time) {
  if (points.size() < 4 || knot_interval <= 0.0) {
    throw std::invalid_argument("Invalid B-spline trajectory.");
  }
  const double duration = (points.size() - 3.0) * knot_interval;
  const double time = std::clamp(trajectory_time, 0.0, duration);
  const double scaled = time / knot_interval;
  const std::size_t segment = std::min(
      static_cast<std::size_t>(std::floor(scaled)), points.size() - 4);
  const double u = time >= duration ? 1.0 : scaled - segment;
  const double u2 = u * u;
  const std::array<double, 4> basis{
      -0.5 * (1.0 - u) * (1.0 - u),
      -2.0 * u + 1.5 * u2,
      0.5 + u - 1.5 * u2,
      0.5 * u2};
  Vec3 result{};
  for (std::size_t i = 0; i < 4; ++i) {
    result = add(result, multiply(points[segment + i], basis[i] / knot_interval));
  }
  return result;
}

Vec3 EgoPlanner::evaluateAcceleration(
    const std::vector<Vec3>& points,
    double knot_interval,
    double trajectory_time) {
  if (points.size() < 4 || knot_interval <= 0.0) {
    throw std::invalid_argument("Invalid B-spline trajectory.");
  }
  const double duration = (points.size() - 3.0) * knot_interval;
  const double time = std::clamp(trajectory_time, 0.0, duration);
  const double scaled = time / knot_interval;
  const std::size_t segment = std::min(
      static_cast<std::size_t>(std::floor(scaled)), points.size() - 4);
  const double u = time >= duration ? 1.0 : scaled - segment;
  const std::array<double, 4> basis{1.0 - u, -2.0 + 3.0 * u,
                                     1.0 - 3.0 * u, u};
  Vec3 result{};
  const double inverse_dt2 = 1.0 / (knot_interval * knot_interval);
  for (std::size_t i = 0; i < 4; ++i) {
    result = add(result, multiply(points[segment + i], basis[i] * inverse_dt2));
  }
  return result;
}

bool EgoPlanner::trajectoryIsFree(
    const GridView& grid,
    const std::vector<Vec3>& points,
    double knot_interval) const {
  const double duration = (points.size() - 3.0) * knot_interval;
  const int count = std::max(
      2, static_cast<int>(std::ceil(duration / config_.collision_check_step)) + 1);
  Index3 previous{};
  bool has_previous = false;
  for (int sample = 0; sample < count; ++sample) {
    const double time = duration * sample / (count - 1.0);
    const Index3 index =
        grid.worldToGrid(evaluateBspline(points, knot_interval, time));
    if (grid.isBlocked(index) ||
        (has_previous && !transitionIsFree(grid, previous, index))) {
      return false;
    }
    previous = index;
    has_previous = true;
  }
  return true;
}

TrajectoryResult EgoPlanner::reboundReplan(
    const GridView& grid,
    const std::vector<Vec3>& seed_path,
    const Vec3& start_velocity,
    const Vec3& start_acceleration,
    const Vec3& target_velocity) const {
  if (seed_path.size() < 2 ||
      norm(subtract(seed_path.front(), seed_path.back())) < 0.05) {
    return {};
  }
  double knot_interval = config_.knot_interval;
  const std::vector<Vec3> point_set = buildInitialPointSet(
      seed_path, start_velocity, start_acceleration, target_velocity,
      knot_interval);
  std::vector<Vec3> control_points = parameterizeToBspline(
      point_set, knot_interval, start_velocity, target_velocity,
      start_acceleration, {0.0, 0.0, 0.0});
  if (control_points.size() <= 2 * kOrder) {
    return {};
  }

  ConstraintList constraints;
  if (!initializeReboundConstraints(grid, control_points, constraints)) {
    return {};
  }
  int rebound_count = 0;
  if (!optimizeRebound(grid, control_points, constraints, knot_interval,
                       rebound_count)) {
    return {false, "rebound_failed", {}, 0.0, rebound_count, false};
  }

  bool time_reallocated = false;
  const double ratio = feasibleRatio(control_points, knot_interval);
  if (ratio > 1.0 + config_.feasibility_tolerance) {
    time_reallocated = true;
    std::vector<Vec3> reference_positions(control_points.size());
    for (std::size_t i = 1; i + 1 < control_points.size(); ++i) {
      reference_positions[i] = multiply(
          add(add(control_points[i - 1], multiply(control_points[i], 4.0)),
              control_points[i + 1]),
          1.0 / 6.0);
    }
    reference_positions.front() = reference_positions[1];
    reference_positions.back() = reference_positions[control_points.size() - 2];
    knot_interval *= ratio * 1.01;
    const std::vector<Vec3> before_refine = control_points;
    optimizeRefine(control_points, reference_positions, knot_interval);
    if (!trajectoryIsFree(grid, control_points, knot_interval)) {
      control_points = before_refine;
    }
  }
  if (!trajectoryIsFree(grid, control_points, knot_interval)) {
    return {false, "collision_after_refine", {}, 0.0, rebound_count,
            time_reallocated};
  }
  return {true,
          time_reallocated ? "rebound_refined" : "rebound_optimized",
          control_points,
          knot_interval,
          rebound_count,
          time_reallocated};
}

}  // namespace native_ego
