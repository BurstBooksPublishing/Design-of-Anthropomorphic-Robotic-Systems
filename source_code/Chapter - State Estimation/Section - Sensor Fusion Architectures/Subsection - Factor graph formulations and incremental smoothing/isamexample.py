import numpy as np
from gtsam import (
    NonlinearFactorGraph, Values, ISAM2, ISAM2Params, Pose3, Vector3, Vector6,
    PriorFactor, BetweenFactor, PreintegratedImuMeasurements, PreintegratedImuFactor,
    noiseModel, Symbol
)

def build_symbol(key_type: str, time_step: int) -> Symbol:
    """Create a GTSAM Symbol for different variable types."""
    type_map = {'P': 0, 'V': 1, 'B': 2, 'F': 3}  # Pose, Velocity, Bias, Foot
    return Symbol(type_map[key_type], time_step)

def incremental_slam_optimization(T0: Pose3, prior_cov: np.ndarray, 
                                preint_meas: PreintegratedImuMeasurements,
                                N: int, k: int, 
                                predicted_pose: Pose3, predicted_vel: Vector3, 
                                predicted_bias: Vector6, foot_cov: np.ndarray,
                                vo_cov: np.ndarray, contact_flag: callable, 
                                vo_measurement_available: callable,
                                measured_foot: Pose3, vo_meas: Pose3) -> Values:
    """
    Perform incremental SLAM optimization using ISAM2 with IMU, vision, and contact constraints.
    """
    # Initialize graph, initial estimates, and ISAM2 optimizer
    graph = NonlinearFactorGraph()
    initial = Values()
    params = ISAM2Params()
    isam = ISAM2(params)

    # Add prior constraint at initial time step
    graph.add(PriorFactor(build_symbol('P', 0), T0, noiseModel.Gaussian.Covariance(prior_cov)))
    initial.insert(build_symbol('P', 0), T0)

    for t in range(N - 1):
        # Add IMU preintegration factor between consecutive time steps
        imu_factor = PreintegratedImuFactor(
            build_symbol('P', t), build_symbol('V', t), build_symbol('B', t),
            build_symbol('P', t + 1), build_symbol('V', t + 1), build_symbol('B', t + 1),
            preint_meas
        )
        graph.add(imu_factor)

        # Add kinematic foot constraint when contact is detected
        if contact_flag(t):
            graph.add(BetweenFactor(
                build_symbol('P', t), build_symbol('F', t), 
                measured_foot, noiseModel.Gaussian.Covariance(foot_cov)
            ))

        # Add vision-based pose-pose constraint when measurement is available
        if vo_measurement_available(t) and t + k < N:
            graph.add(BetweenFactor(
                build_symbol('P', t), build_symbol('P', t + k),
                vo_meas, noiseModel.Gaussian.Covariance(vo_cov)
            ))

        # Insert initial estimates for new state variables
        initial.insert(build_symbol('P', t + 1), predicted_pose)
        initial.insert(build_symbol('V', t + 1), predicted_vel)
        initial.insert(build_symbol('B', t + 1), predicted_bias)

        # Perform incremental optimization update
        isam.update(graph, initial)
        result = isam.calculateEstimate()

        # Clear factor graph and initial estimates for next iteration
        graph.resize(0)
        initial.clear()

    return result