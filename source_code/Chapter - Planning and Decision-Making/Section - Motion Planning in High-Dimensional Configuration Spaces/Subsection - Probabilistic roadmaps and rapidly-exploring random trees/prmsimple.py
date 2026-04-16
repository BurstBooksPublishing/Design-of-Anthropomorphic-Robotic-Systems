\begin{lstlisting}[language=Python,caption={Minimal RRT* iteration with radius based neighbor and rewiring (core logic only).},label={lst:rrtstar}]

import numpy as np
from typing import List, Tuple, Optional, Any

def rrt_star_iter(tree: List[dict], x_rand: np.ndarray, r_n: float) -> None:
    """
    Single RRT* iteration: extend tree towards random sample with optimal rewiring.
    
    Args:
        tree: List of nodes, each with 'state', 'parent', and 'cost' keys
        x_rand: Random state sample
        r_n: Neighborhood radius for rewiring
    """
    # Find nearest node and attempt to extend toward random sample
    x_nearest = nearest(tree, x_rand)
    x_new = steer(x_nearest, x_rand)
    
    if not collision_free(x_nearest['state'], x_new['state']):
        return
    
    # Find nearby nodes within radius for potential parent selection
    X_near = near(tree, x_new['state'], r_n)
    
    # Select optimal parent from nearby nodes
    x_min = x_nearest
    c_min = x_nearest['cost'] + cost_edge(x_nearest['state'], x_new['state'])
    
    for x_near in X_near:
        c = x_near['cost'] + cost_edge(x_near['state'], x_new['state'])
        if c < c_min and collision_free(x_near['state'], x_new['state']):
            x_min, c_min = x_near, c
    
    # Add new node with optimal parent
    add_node(tree, x_new, parent=x_min, cost=c_min)
    
    # Rewire nearby nodes if path through new node is more efficient
    for x_near in X_near:
        c_through = x_new['cost'] + cost_edge(x_new['state'], x_near['state'])
        if c_through < x_near['cost'] and collision_free(x_new['state'], x_near['state']):
            rewire(tree, x_near, new_parent=x_new)

# Helper functions (stub implementations)
def nearest(tree: List[dict], x_rand: np.ndarray) -> dict:
    # Return node with minimum Euclidean distance to x_rand
    pass

def steer(x_from: dict, x_to: np.ndarray) -> dict:
    # Return new node state moved from x_from toward x_to
    pass

def collision_free(state1: np.ndarray, state2: np.ndarray) -> bool:
    # Check if path between states is collision-free
    pass

def near(tree: List[dict], x_new: np.ndarray, r_n: float) -> List[dict]:
    # Return nodes within radius r_n of x_new
    pass

def cost_edge(state1: np.ndarray, state2: np.ndarray) -> float:
    # Return cost of edge between two states
    pass

def add_node(tree: List[dict], x_new: dict, parent: dict, cost: float) -> None:
    # Add new node to tree with parent and cost
    pass

def rewire(tree: List[dict], node: dict, new_parent: dict) -> None:
    # Update node's parent and cost in tree
    pass