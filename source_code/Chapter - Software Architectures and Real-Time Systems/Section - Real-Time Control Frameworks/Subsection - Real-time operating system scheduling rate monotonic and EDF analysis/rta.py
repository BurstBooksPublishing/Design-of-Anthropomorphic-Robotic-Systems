import math
from typing import List, Tuple

# Task format: (computation_time, period, deadline)
tasks = [  # (C_ms, T_ms, D_ms)
    (0.1, 1.0, 1.0),
    (0.6, 2.0, 2.0),
    (1.5, 5.0, 5.0),
    (6.0, 33.333, 33.333),
]

# Response-time analysis for RM (sorted by period)
tasks_sorted = sorted(tasks, key=lambda x: x[1])

def rta(task_idx: int) -> float:
    """Calculate response time for task at given index using RTA for RM."""
    C_i, T_i, D_i = tasks_sorted[task_idx]
    R = C_i
    
    while True:
        interference = 0.0
        # Sum interference from all higher-priority tasks
        for j in range(task_idx):
            C_j, T_j, _ = tasks_sorted[j]
            interference += (math.ceil(R / T_j) * C_j)
        
        R_next = C_i + interference
        
        if R_next == R: 
            break
            
        # Deadline miss detected
        if R_next > D_i: 
            return float('inf')
            
        R = R_next
    
    return R

def dbf(t: float) -> float:
    """Calculate demand bound function for EDF scheduling."""
    s = 0.0
    for C, T, D in tasks:
        # Number of jobs released within time interval [0, t]
        k = max(0, math.floor((t - D) / T) + 1)
        s += k * C
    return s

# Example usage:
# results = [(i, rta(i)) for i in range(len(tasks_sorted))]
# print("RTA Results:", results)
# print("DBF at t=10:", dbf(10.0))