#pragma once
#include <Eigen/Dense>
#include <deque>
#include <fstream>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "point_types.hpp"
#include "so3_math.h"
using namespace std;
using namespace Eigen;
using V3D=Eigen::Vector3d;
using M3D=Eigen::Matrix3d;
inline const V3D Zero3d=V3D::Zero();
inline const M3D Eye3d=M3D::Identity();
#define MD(a,b) Eigen::Matrix<double,a,b>
#define G_m_s2 9.81
#define NUM_MATCH_POINTS 5
#define VEC_FROM_ARRAY(v) v[0],v[1],v[2]
#define MAT_FROM_ARRAY(v) v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]
#define ROS_ASSERT(v) assert(v)
#define ROS_INFO(...) ((void)0)
#define MARSIM 4
inline double omp_get_wtime() {return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();}
// Plain C++ containers match the fields consumed by official ImuProcess;
// no ROS node, clock or transport is linked.
namespace sensor_msgs {
struct Imu {
 using ConstPtr=std::shared_ptr<const Imu>;
 struct Stamp {double value=0.; double toSec() const {return value;}};
 struct Header {Stamp stamp;} header;
 struct Vector {double x=0,y=0,z=0;};
 Vector linear_acceleration,angular_velocity;
};
using ImuConstPtr=Imu::ConstPtr;
}
struct Pose6D {double offset_time=0,acc[3]{},gyr[3]{},vel[3]{},pos[3]{},rot[9]{};};
struct MeasureGroup {
 double lidar_beg_time=0, lidar_end_time=0;
 PointCloudXYZI::Ptr lidar=std::make_shared<PointCloudXYZI>();
 std::deque<sensor_msgs::Imu::ConstPtr> imu;
};
#include "geometry.hpp"
