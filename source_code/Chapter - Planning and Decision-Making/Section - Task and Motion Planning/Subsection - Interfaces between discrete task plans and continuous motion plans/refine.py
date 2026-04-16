from typing import List, Tuple, Any, Optional, Union

def refine_plan(plan: List[Any], x0: Any) -> Tuple[bool, Union[int, List[Any]]]:
    """
    Refines a sequence of planned actions into executable trajectories.
    
    Args:
        plan: List of high-level actions to execute
        x0: Initial state of the system
        
    Returns:
        Tuple of (success_flag, result) where result is either:
        - failure index (int) if planning fails
        - list of trajectories (List) if successful
    """
    trajs = []
    state = x0
    
    for i, action in enumerate(plan):
        # Perform fast geometric feasibility check before expensive optimization
        if not quick_check(action, state):
            return False, i  # Early termination on geometric infeasibility
            
        # Compute detailed trajectory using optimal control solver
        sol = motion_planner(action, state)
        if sol is None:
            return False, i  # Planning failed for this action
            
        trajs.append(sol)
        state = sol.end_state
        
    return True, trajs