# theta: current joints (n,) ; T_des: desired SE(3) as 4x4
for k in range(max_iter):
    T = forward_kinematics(theta)                # 4x4
    err = se3_log(np.linalg.inv(T) @ T_des)      # 6-vector error
    if np.linalg.norm(err) < tol: break
    J = body_jacobian(theta)                     # 6xn
    # solve (J^T J + lambda I) delta = -J^T err  (LM damping)
    A = J.T @ J + lam * np.eye(len(theta))
    b = -J.T @ err
    delta = np.linalg.solve(A, b)                # update
    theta = theta + delta
    # optional null-space projection for secondary task