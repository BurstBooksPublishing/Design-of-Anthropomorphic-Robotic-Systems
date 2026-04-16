import math
from typing import Union
from dds import DataWriterQoS, Writer

def delay_bound(burst_bytes: Union[int, float], rate_bps: Union[int, float], 
                latency_s: Union[int, float]) -> float:
    """Calculate delay bound using burst size, rate, and latency."""
    if rate_bps <= 0:
        raise ValueError("Rate must be positive")
    if any(param < 0 for param in [burst_bytes, latency_s]):
        raise ValueError("Burst size and latency must be non-negative")
    
    transmission_time = (burst_bytes * 8) / rate_bps
    return transmission_time + latency_s

def configure_dds_qos() -> DataWriterQoS:
    """Configure DDS QoS for real-time control with bounded latency."""
    qos = DataWriterQoS()
    qos.reliability = 'RELIABLE'
    qos.history_kind = 'KEEP_LAST'
    qos.history_depth = 1
    qos.deadline = 0.001
    qos.lifespan = 0.005
    qos.transport_priority = 10
    return qos

def verify_timing_constraints():
    """Verify timing constraints are met for real-time operation."""
    burst_bytes = 4 * 64
    rate_bps = 1e6
    latency_s = 0.0002
    processing_s = 0.0001
    deadline_s = 0.001
    
    delay = processing_s + delay_bound(burst_bytes, rate_bps, latency_s)
    if delay > deadline_s:
        raise RuntimeError(f"Deadline miss ({delay:.6f}s > {deadline_s}s): "
                         "increase rate or reduce burst size")

# Main execution
verify_timing_constraints()
qos_config = configure_dds_qos()
writer: Writer = create_writer(topic, qos=qos_config)