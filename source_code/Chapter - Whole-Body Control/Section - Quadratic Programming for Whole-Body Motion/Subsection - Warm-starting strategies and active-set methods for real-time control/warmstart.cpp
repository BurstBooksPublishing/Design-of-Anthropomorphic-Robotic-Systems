cpp
#include <Eigen/Dense>
#include <Eigen/LU>
#include <cassert>

/**
 * Solves the KKT system for constrained optimization problems.
 * 
 * @param H Cost Hessian matrix (n x n)
 * @param A Active constraint matrix (m x n)
 * @param rhs Right-hand side vector [-gradient; constraint_rhs] (n+m x 1)
 * @param prevL Previous LDLT factorization for warm start (optional)
 * @return Solution vector [x; lambda] where x is primal variables and lambda is dual variables
 */
Eigen::VectorXd solveKKT_warm(const Eigen::MatrixXd& H,
                              const Eigen::MatrixXd& A,
                              const Eigen::VectorXd& rhs,
                              Eigen::LDLT<Eigen::MatrixXd>* prevL = nullptr) {
    const int n = H.rows();
    const int m = A.rows();
    
    // Validate input dimensions
    assert(H.cols() == n);
    assert(A.cols() == n);
    assert(rhs.size() == n + m);
    
    // Construct KKT matrix
    Eigen::MatrixXd K(n + m, n + m);
    K.topLeftCorner(n, n) = H;
    K.topRightCorner(n, m) = A.transpose();
    K.bottomLeftCorner(m, n) = A;
    K.bottomRightCorner(m, m).setZero();
    
    // Use previous factorization if available and compatible
    if (prevL && prevL->isInitialized() && prevL->matrixLDLT().rows() == n + m) {
        // TODO: Implement low-rank update/downdate logic
        // For now, fall through to re-factorization
    }
    
    // Factorize and solve KKT system
    Eigen::LDLT<Eigen::MatrixXd> L(K);
    return L.solve(rhs);
}