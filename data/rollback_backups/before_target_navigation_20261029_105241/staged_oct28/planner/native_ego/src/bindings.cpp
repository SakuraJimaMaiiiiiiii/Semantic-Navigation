#include "native_ego/ego_planner.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace native_ego {
namespace {

using UInt8Array = py::array_t<
    std::uint8_t,
    py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<
    double,
    py::array::c_style | py::array::forcecast>;

template <typename Value>
void assignIfPresent(
    const py::dict& values,
    const char* name,
    Value& destination) {
  if (values.contains(name)) {
    destination = py::cast<Value>(values[name]);
  }
}

PlannerConfig parseConfig(const py::dict& values) {
  PlannerConfig config;
  assignIfPresent(values, "control_point_spacing", config.control_point_spacing);
  assignIfPresent(values, "clearance", config.clearance);
  assignIfPresent(values, "knot_interval", config.knot_interval);
  assignIfPresent(values, "max_velocity", config.max_velocity);
  assignIfPresent(values, "max_acceleration", config.max_acceleration);
  assignIfPresent(
      values, "feasibility_tolerance", config.feasibility_tolerance);
  assignIfPresent(
      values, "optimization_iterations", config.optimization_iterations);
  assignIfPresent(values, "lbfgs_memory", config.lbfgs_memory);
  assignIfPresent(
      values, "rebound_max_restarts", config.rebound_max_restarts);
  assignIfPresent(values, "smoothness_weight", config.smoothness_weight);
  assignIfPresent(values, "collision_weight", config.collision_weight);
  assignIfPresent(values, "feasibility_weight", config.feasibility_weight);
  assignIfPresent(values, "fitness_weight", config.fitness_weight);
  assignIfPresent(values, "collision_check_step", config.collision_check_step);
  assignIfPresent(values, "max_search_nodes", config.max_search_nodes);
  return config;
}

Vec3 readVec3(const DoubleArray& values, const char* name) {
  const auto buffer = values.request();
  if (buffer.ndim != 1 || buffer.shape[0] != 3) {
    throw std::invalid_argument(std::string(name) + " must have shape (3,).");
  }
  const auto* data = static_cast<const double*>(buffer.ptr);
  return {data[0], data[1], data[2]};
}

std::vector<Vec3> readPoints(const DoubleArray& values) {
  const auto buffer = values.request();
  if (buffer.ndim != 2 || buffer.shape[1] != 3) {
    throw std::invalid_argument("seed_path must have shape (N, 3).");
  }
  const auto* data = static_cast<const double*>(buffer.ptr);
  std::vector<Vec3> points;
  points.reserve(static_cast<std::size_t>(buffer.shape[0]));
  for (py::ssize_t row = 0; row < buffer.shape[0]; ++row) {
    const std::size_t offset = static_cast<std::size_t>(row) * 3;
    points.push_back({data[offset], data[offset + 1], data[offset + 2]});
  }
  return points;
}

py::array_t<double> pointsToArray(const std::vector<Vec3>& points) {
  py::array_t<double> result(
      {static_cast<py::ssize_t>(points.size()), py::ssize_t{3}});
  auto output = result.mutable_unchecked<2>();
  for (py::ssize_t row = 0;
       row < static_cast<py::ssize_t>(points.size());
       ++row) {
    for (py::ssize_t axis = 0; axis < 3; ++axis) {
      output(row, axis) = points[static_cast<std::size_t>(row)][axis];
    }
  }
  return result;
}

GridView makeGrid(
    const UInt8Array& blocked,
    const DoubleArray& origin,
    double resolution) {
  const auto blocked_buffer = blocked.request();
  if (blocked_buffer.ndim != 3) {
    throw std::invalid_argument("blocked must be a three-dimensional array.");
  }
  if (resolution <= 0.0) {
    throw std::invalid_argument("resolution must be greater than zero.");
  }
  GridView grid;
  grid.blocked = static_cast<const std::uint8_t*>(blocked_buffer.ptr);
  grid.shape = {
      static_cast<int>(blocked_buffer.shape[0]),
      static_cast<int>(blocked_buffer.shape[1]),
      static_cast<int>(blocked_buffer.shape[2]),
  };
  grid.origin = readVec3(origin, "origin");
  grid.resolution = resolution;
  return grid;
}

class NativeEgoPlanner {
 public:
  explicit NativeEgoPlanner(const py::dict& config)
      : planner_(parseConfig(config)) {}

  py::dict reboundReplan(
      const DoubleArray& seed_path,
      const UInt8Array& blocked,
      const DoubleArray& origin,
      double resolution,
      const DoubleArray& start_velocity,
      const DoubleArray& start_acceleration,
      const DoubleArray& target_velocity) const {
    const GridView grid = makeGrid(blocked, origin, resolution);
    const std::vector<Vec3> seed = readPoints(seed_path);
    const Vec3 start_velocity_value = readVec3(
        start_velocity, "start_velocity");
    const Vec3 start_acceleration_value = readVec3(
        start_acceleration, "start_acceleration");
    const Vec3 target_velocity_value = readVec3(
        target_velocity, "target_velocity");
    TrajectoryResult result;
    {
      py::gil_scoped_release release;
      result = planner_.reboundReplan(
          grid,
          seed,
          start_velocity_value,
          start_acceleration_value,
          target_velocity_value);
    }
    py::dict output;
    output["success"] = result.success;
    output["status"] = result.status;
    output["control_points"] = pointsToArray(result.control_points);
    output["knot_interval"] = result.knot_interval;
    output["rebound_count"] = result.rebound_count;
    output["time_reallocated"] = result.time_reallocated;
    return output;
  }

 private:
  EgoPlanner planner_;
};

}  // namespace
}  // namespace native_ego

PYBIND11_MODULE(_native_ego, module) {
  module.doc() = "ROS-free adapter of the EGO-Planner core pipeline";
  py::class_<native_ego::NativeEgoPlanner>(module, "NativeEgoPlanner")
      .def(py::init<const py::dict&>(), py::arg("config") = py::dict())
      .def(
          "rebound_replan",
          &native_ego::NativeEgoPlanner::reboundReplan,
          py::arg("seed_path"),
          py::arg("blocked"),
          py::arg("origin"),
          py::arg("resolution"),
          py::arg("start_velocity"),
          py::arg("start_acceleration"),
          py::arg("target_velocity"));
}
