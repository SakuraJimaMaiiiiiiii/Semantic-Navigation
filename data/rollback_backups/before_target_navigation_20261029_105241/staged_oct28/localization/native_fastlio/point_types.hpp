#pragma once
#include <Eigen/Core>
#include <vector>
#include <memory>
struct PointType { float x=0,y=0,z=0,intensity=0,curvature=0; };
using PointVector = std::vector<PointType,Eigen::aligned_allocator<PointType>>;
struct PointCloudXYZI {
    using Ptr = std::shared_ptr<PointCloudXYZI>;
    PointVector points;
    size_t size() const {return points.size();}
    bool empty() const {return points.empty();}
};
