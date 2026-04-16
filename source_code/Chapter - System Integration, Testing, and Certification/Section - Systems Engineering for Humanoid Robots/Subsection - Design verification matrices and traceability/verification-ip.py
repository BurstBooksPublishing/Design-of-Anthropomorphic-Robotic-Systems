import pulp
from typing import List, Tuple

def solve_test_selection(T: List[List[int]], c: List[int]) -> Tuple[List[int], float]:
    """
    Solve the test selection problem to minimize cost while ensuring full coverage.
    
    Args:
        T: Coverage matrix where T[i][j] = 1 if test j covers requirement i
        c: Cost vector where c[j] is the cost of test j
        
    Returns:
        Tuple of (selected_tests, total_cost)
    """
    m, n = len(T), len(T[0]) if T else 0
    
    # Create optimization problem
    prob = pulp.LpProblem("min_verification", pulp.LpMinimize)
    
    # Decision variables: x[j] = 1 if test j is selected
    x = [pulp.LpVariable(f"x{j}", cat="Binary") for j in range(n)]
    
    # Objective: minimize total cost
    prob += pulp.lpSum(c[j] * x[j] for j in range(n))
    
    # Coverage constraints: each requirement must be covered by at least one test
    for i in range(m):
        prob += pulp.lpSum(T[i][j] * x[j] for j in range(n)) >= 1
    
    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract solution
    selected_tests = [j for j in range(n) if pulp.value(x[j]) > 0.5]
    total_cost = pulp.value(prob.objective) if prob.status == pulp.LpStatusOptimal else float('inf')
    
    return selected_tests, total_cost

# Example usage
if __name__ == "__main__":
    T = [[1,1,0,0,0],
         [0,1,1,0,0],
         [0,0,0,1,0],
         [0,0,0,0,1]]
    c = [2,5,6,10,3]
    
    selected, cost = solve_test_selection(T, c)
    print(f"Selected tests: {selected}")
    print(f"Total cost: {cost}")