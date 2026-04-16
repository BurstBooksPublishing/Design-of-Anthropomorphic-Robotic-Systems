from collections import deque, defaultdict
from typing import Dict, List, Tuple, Any

def topological_order(G: Dict[Any, Dict]) -> List[Any]:
    """Return topological ordering of graph nodes."""
    in_degree = {v: 0 for v in G}
    for v in G:
        for succ in G[v]['succ']:
            in_degree[succ] += 1
    
    queue = deque([v for v in G if in_degree[v] == 0])
    result = []
    
    while queue:
        v = queue.popleft()
        result.append(v)
        for w in G[v]['succ']:
            in_degree[w] -= 1
            if in_degree[w] == 0:
                queue.append(w)
    
    return result

def compute_W_L(G: Dict[Any, Dict]) -> Tuple[float, float]:
    """Compute total work and critical path length."""
    W = sum(G[v]['cost'] for v in G)
    
    # Longest path calculation (cost + communication delay when crossing processors)
    topo = topological_order(G)
    L = {v: G[v]['cost'] for v in G}
    
    for v in topo:
        for w in G[v]['succ']:
            # Communication cost is added when tasks are on different processors
            comm_cost = G[v]['comm'].get(w, 0)
            L[w] = max(L[w], L[v] + G[w]['cost'] + comm_cost)
    
    return W, max(L.values())

def list_schedule(G: Dict[Any, Dict], P: int) -> Tuple[float, List[Tuple]]:
    """Schedule tasks using list scheduling algorithm."""
    # Initialize simulation state
    time = 0.0
    in_degree = {v: 0 for v in G}
    for v in G:
        for succ in G[v]['succ']:
            in_degree[succ] += 1
    
    ready = deque([v for v in G if in_degree[v] == 0])
    busy = []  # (end_time, proc_id, task_id)
    proc_free = list(range(P))
    events = []
    
    while ready or busy:
        # Assign ready tasks to available processors
        while proc_free and ready:
            v = ready.popleft()
            p = proc_free.pop()
            end = time + G[v]['cost']
            busy.append((end, p, v))
            events.append((time, 'start', v, p))
        
        # Advance time to next task completion
        if busy:
            busy.sort()
            end, p, v = busy.pop(0)
            time = end
            proc_free.append(p)
            events.append((time, 'finish', v, p))
            
            # Check for newly ready successor tasks
            for w in G[v]['succ']:
                in_degree[w] -= 1
                if in_degree[w] == 0:
                    ready.append(w)
    
    return time, events