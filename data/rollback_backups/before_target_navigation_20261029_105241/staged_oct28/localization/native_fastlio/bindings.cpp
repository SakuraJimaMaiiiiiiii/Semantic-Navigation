// ROS-free host for pinned FAST-LIO2 / IKFoM / ikd-Tree.
// Measurement equations follow FAST_LIO src/laserMapping.cpp h_share_model.
// Upstream licenses are retained in third_party; this adapter is GPL-2.0.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <map>
#include <array>
#include "transport.hpp"
#include "IMU_Processing.hpp"
#include "ikd_Tree.h"
namespace py=pybind11;
using Filter=esekfom::esekf<state_ikfom,12,input_ikfom>;
class FastLio {
    Filter filter;
    ImuProcess imu;
    KD_TREE<PointType> tree;
    PointVector scan;
    double last_time=-1.;
    double last_imu_time=-1.;
    double voxel;
    bool snapshot;
    int effective=0;
    double residual=0.;
    static thread_local FastLio* active;
    static void measurement(state_ikfom& s, esekfom::dyn_share_datastruct<double>& data) {
        active->observe(s,data);
    }
    void observe(state_ikfom& s, esekfom::dyn_share_datastruct<double>& data) {
        std::vector<Eigen::Matrix<double,1,12>> rows;
        std::vector<double> errors;
        residual=0.;
        for(const auto& p:scan) {
            V3D body(p.x,p.y,p.z);
            V3D in_imu=s.offset_R_L_I*body+s.offset_T_L_I;
            V3D world=s.rot*in_imu+s.pos;
            PointType query;query.x=world.x();query.y=world.y();query.z=world.z();
            PointVector neighbors;std::vector<float> distances;
            tree.Nearest_Search(query,5,neighbors,distances);
            if(neighbors.size()!=5 || distances.back()>5.) continue;
            Eigen::Vector4d plane;
            if(!esti_plane(plane,neighbors,.1) || !plane.allFinite()) continue;
            double error=plane.head<3>().dot(world)+plane[3];
            if(body.norm()<1e-6 || 1.-.9*std::abs(error)/std::sqrt(body.norm())<=.9) continue;
            V3D C=s.rot.conjugate()*plane.head<3>();
            V3D A=skew_sym_mat(in_imu)*C;
            Eigen::Matrix<double,1,12> row=Eigen::Matrix<double,1,12>::Zero();
            row.head<3>()=plane.head<3>().transpose();row.segment<3>(3)=A.transpose();
            // Calibrated extrinsics are fixed, as extrinsic_est_en=false upstream.
            rows.push_back(row);errors.push_back(-error);residual+=std::abs(error);
        }
        effective=int(rows.size());
        if(effective<12) {data.valid=false;return;}
        data.h_x.resize(effective,12);data.h.resize(effective);
        for(int i=0;i<effective;++i){data.h_x.row(i)=rows[i];data.h[i]=errors[i];}
        residual/=effective;
    }
public:
    FastLio(bool instantaneous, double voxel_size,
            const Eigen::Matrix3d& lidar_to_imu_rotation,const V3D& lidar_to_imu_translation,
            double gyro_variance,double accel_variance,double gyro_bias_variance,double accel_bias_variance)
        :voxel(voxel_size),snapshot(instantaneous) {
        if(!std::isfinite(voxel)||voxel<=0) throw std::invalid_argument("voxel_size must be positive");
        if(!lidar_to_imu_rotation.allFinite() || !lidar_to_imu_translation.allFinite()
          || (lidar_to_imu_rotation.transpose()*lidar_to_imu_rotation-M3D::Identity()).norm()>1e-6
          || std::abs(lidar_to_imu_rotation.determinant()-1)>1e-6)
            throw std::invalid_argument("Invalid LiDAR-to-IMU rigid transform");
        for(double q:{gyro_variance,accel_variance,gyro_bias_variance,accel_bias_variance})
            if(!std::isfinite(q)||q<=0) throw std::invalid_argument("IMU variances must be positive");
        double epsilon[23];std::fill(epsilon,epsilon+23,.001);
        filter.init_dyn_share(get_f,df_dx,df_dw,measurement,4,epsilon);
        imu.lidar_type=snapshot?MARSIM:1;
        imu.set_extrinsic(lidar_to_imu_translation,lidar_to_imu_rotation);
        imu.set_gyr_cov(V3D::Constant(gyro_variance));imu.set_acc_cov(V3D::Constant(accel_variance));
        imu.set_gyr_bias_cov(V3D::Constant(gyro_bias_variance));imu.set_acc_bias_cov(V3D::Constant(accel_bias_variance));
        tree.set_downsample_param(float(voxel));
    }
    py::dict process(double start,const Eigen::MatrixXd& points,const Eigen::MatrixXd& samples) {
        // samples: source seconds, FRD specific force m/s², FRD gyro rad/s.
        // points: LiDAR XYZ m, seconds from scan start (zero only for snapshots).
        if(!std::isfinite(start)||start<=last_time || points.cols()!=4||points.rows()<12
            || samples.cols()!=7||samples.rows()<2 || !points.allFinite() || !samples.allFinite())
            throw std::invalid_argument("Expected ordered scan, finite Nx4 points and Mx7 raw IMU");
        double duration=points.col(3).maxCoeff();
        if(points.col(3).minCoeff()<0 || duration>1 || (snapshot && duration!=0))
            throw std::invalid_argument("Invalid point times for selected scan model");
        double end=start+duration;
        for(int i=0;i<samples.rows();++i)
            if(samples(i,0)<=last_imu_time || (i && samples(i,0)<=samples(i-1,0)))
                throw std::invalid_argument("IMU source time must be strictly increasing across calls");
        if(samples(samples.rows()-1,0)>end+1e-9 || end-samples(samples.rows()-1,0)>.03)
            throw std::invalid_argument("IMU must cover scan end within 30 ms, without future samples");
        if(last_time>=0 && samples(0,0)-last_imu_time>.1)
            throw std::invalid_argument("IMU gap exceeds 100 ms");
        for(int i=1;i<samples.rows();++i)
            if(samples(i,0)-samples(i-1,0)>.1) throw std::invalid_argument("IMU gap exceeds 100 ms");
        MeasureGroup group;group.lidar_beg_time=start;group.lidar_end_time=end;
        for(int i=0;i<points.rows();++i) {
            PointType p;p.x=points(i,0);p.y=points(i,1);p.z=points(i,2);p.curvature=points(i,3)*1000.;
            group.lidar->points.push_back(p);
        }
        for(int i=0;i<samples.rows();++i) {
            auto m=std::make_shared<sensor_msgs::Imu>();m->header.stamp.value=samples(i,0);
            m->linear_acceleration={samples(i,1),samples(i,2),samples(i,3)};
            m->angular_velocity={samples(i,4),samples(i,5),samples(i,6)};group.imu.push_back(m);
        }
        auto cloud=std::make_shared<PointCloudXYZI>();
        {py::gil_scoped_release unlocked;imu.Process(group,filter,cloud);}
        last_time=end;last_imu_time=samples(samples.rows()-1,0);
        py::dict result;result["timestamp"]=end;
        if(cloud->empty()){result["status"]="initializing";return result;}
        // Voxel centroid sampling replaces only PCL's VoxelGrid container operation.
        std::map<std::array<int,3>,std::pair<V3D,int>> bins;
        for(const auto& p:cloud->points) {
            std::array<int,3> key{int(std::floor(p.x/voxel)),int(std::floor(p.y/voxel)),int(std::floor(p.z/voxel))};
            auto it=bins.try_emplace(key,V3D::Zero(),0).first;
            it->second.first+=V3D(p.x,p.y,p.z);++it->second.second;
        }
        scan.clear();
        for(auto& item:bins){V3D p=item.second.first/item.second.second;PointType q;q.x=p.x();q.y=p.y();q.z=p.z();scan.push_back(q);}
        bool built=tree.Root_Node!=nullptr;
        if(built){py::gil_scoped_release unlocked;active=this;double solve=0.;filter.update_iterated_dyn_share_modified(.001,solve);active=nullptr;}
        auto state=filter.get_x();
        Eigen::Matrix4d pose=Eigen::Matrix4d::Identity();pose.block<3,3>(0,0)=state.rot.toRotationMatrix();pose.block<3,1>(0,3)=state.pos;
        if(!pose.allFinite()) throw std::runtime_error("FAST-LIO produced nonfinite pose");
        PointVector world;
        Eigen::MatrixXd output(scan.size(),3);
        for(size_t i=0;i<scan.size();++i){auto q=scan[i];V3D p=state.rot*(state.offset_R_L_I*V3D(q.x,q.y,q.z)+state.offset_T_L_I)+state.pos;
            q.x=p.x();q.y=p.y();q.z=p.z();world.push_back(q);output.row(i)=p.transpose();}
        {py::gil_scoped_release unlocked;
            if(!built) tree.Build(world);
            else if(effective>=12) tree.Add_Points(world,true);
        }
        result["status"]=!built?"map_initialized":effective>=12?"tracking":"insufficient_correspondences";
        result["pose"]=pose;result["points_world"]=output;result["velocity"]=Eigen::Vector3d(state.vel);
        result["effective_points"]=effective;result["mean_residual_m"]=residual;result["map_points"]=tree.size();
        result["covariance"]=Eigen::MatrixXd(filter.get_P());return result;
    }
};
thread_local FastLio* FastLio::active=nullptr;
PYBIND11_MODULE(_native_fastlio,m) {
    m.attr("UPSTREAM_COMMIT")="7cc4175de6f8ba2edf34bab02a42195b141027e9";
    m.attr("ALGORITHM_VERSION")="fastlio2_ros_free_fixed_extrinsics_v1";
    py::class_<FastLio>(m,"Estimator")
      .def(py::init<bool,double,const M3D&,const V3D&,double,double,double,double>(),
        py::arg("instantaneous"),py::arg("voxel_size")=.3,
        py::arg("lidar_to_imu_rotation")=M3D::Identity(),py::arg("lidar_to_imu_translation")=V3D::Zero(),
        py::arg("gyro_variance")=.1,py::arg("accel_variance")=.1,
        py::arg("gyro_bias_variance")=.0001,py::arg("accel_bias_variance")=.0001)
      .def("process",&FastLio::process);
}
