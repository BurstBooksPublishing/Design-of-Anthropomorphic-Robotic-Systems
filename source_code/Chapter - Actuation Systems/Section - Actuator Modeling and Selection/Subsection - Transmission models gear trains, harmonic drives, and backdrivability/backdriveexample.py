import math

def calculate_joint_dynamics(
    joint_inertia: float = 0.05,      # kg*m^2
    motor_inertia: float = 1e-4,      # kg*m^2
    stiffness: float = 1e4,           # N*m/rad
    gear_ratio: float = 100,          # unitless
    efficiency: float = 0.9,          # unitless
    motor_max_torque: float = 2.0,    # N*m
    joint_friction: float = 0.5       # N*m
) -> tuple[float, float]:
    """
    Calculate joint dynamics parameters for robotic system analysis.
    
    Returns:
        tuple: (natural_frequency, backdrivability_index)
    """
    # Reflect joint inertia to motor side through gear ratio
    reflected_joint_inertia = joint_inertia / (gear_ratio ** 2)
    
    # Calculate system natural frequency
    inertia_sum = (1.0 / motor_inertia) + (1.0 / reflected_joint_inertia)
    natural_frequency = math.sqrt(stiffness * inertia_sum)
    
    # Calculate backdrivability index (torque ratio)
    backdrivability_index = (efficiency * gear_ratio * motor_max_torque) / joint_friction
    
    return natural_frequency, backdrivability_index

# Calculate and display results
omega_n, B_index = calculate_joint_dynamics()
print(f"Natural frequency: {omega_n:.2f} rad/s")
print(f"Backdrivability index: {B_index:.2f}")