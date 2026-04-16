cpp
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <Eigen/Core>

using Pose = gtsam::Pose3;
using Vector3 = Eigen::Vector3d;
using namespace gtsam;

// Add IMU preintegration factors
for (const auto& imu_pair : imu_pairs) {
    graph.add(BetweenFactor<Pose3>(Symbol('x', imu_pair.i), Symbol('x', imu_pair.j), 
                                   imu_pair.preintegrated_measurement, imu_cov));
}

// Add vision reprojection factors
for (const auto& meas : vision_meas) {
    graph.add(GenericProjectionFactor<Pose3, Point3>(
        meas.measurement, meas.noise, Symbol('x', meas.pose_id), Symbol('l', meas.landmark_id),
        meas.calibration));
}

// Add LiDAR ICP factors
for (const auto& scan_pair : lidar_pairs) {
    graph.add(ICPFactor(Symbol('x', scan_pair.source_id), Symbol('x', scan_pair.target_id),
                        scan_pair.point_correspondences, scan_pair.normal_vectors, icp_cov));
}

// Optimize the factor graph
Values result = LevenbergMarquardtOptimizer(graph, initial_estimate).optimize();