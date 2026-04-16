from typing import NamedTuple, Set, Tuple, List

class Contract(NamedTuple):
    """Represents an assume-guarantee contract with abstract predicates."""
    assume: Set[str]  # Preconditions that must be satisfied
    guarantee: Set[str]  # Postconditions that are ensured

class PeriodicTask(NamedTuple):
    """Represents a periodic real-time task."""
    C: float  # Worst-case execution time
    T: float  # Period
    D: float  # Relative deadline

def compatible(c1: Contract, c2: Contract) -> bool:
    """Check if two contracts are compatible (c1.guarantee implies c2.assume)."""
    return c2.assume.issubset(c1.guarantee)

def edf_schedulable(tasks: List[PeriodicTask]) -> Tuple[bool, float]:
    """Check EDF schedulability using utilization bound test."""
    utilization = sum(task.C / task.T for task in tasks)
    return utilization <= 1.0, utilization

# Pipeline composition example
sensor = Contract(assume=set(), guarantee={'imu_rate_1kHz'})
estimator = Contract(assume={'imu_rate_1kHz'}, guarantee={'state_1kHz'})
controller = Contract(assume={'state_1kHz'}, guarantee={'torque_1kHz'})
actuator = Contract(assume={'torque_1kHz'}, guarantee=set())

print(compatible(sensor, estimator), compatible(estimator, controller))  # True True

# Task set schedulability example
tasks = [
    PeriodicTask(0.05, 1.0, 1.0),
    PeriodicTask(0.20, 1.0, 1.0),
    PeriodicTask(0.30, 1.0, 1.0),
    PeriodicTask(0.10, 1.0, 1.0)
]
print(edf_schedulable(tasks))  # (True, 0.65)