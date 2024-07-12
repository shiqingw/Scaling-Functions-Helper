#include "rimonMethod.hpp"

xt::xtensor<double, 1> rimonMethod(const xt::xtensor<double, 2>& A, const xt::xtensor<double, 1>& a, const xt::xtensor<double, 2>& B, const xt::xtensor<double, 1>& b) {
    int nv = A.shape()[0];

    xt::xtensor<double, 2> A_sqrt = xt::linalg::cholesky(A); 
    xt::xtensor<double, 2> A_sqrt_inv = xt::linalg::inv(A_sqrt);
    xt::xtensor<double, 2> C = xt::linalg::dot(xt::linalg::dot(A_sqrt_inv, B), xt::transpose(A_sqrt_inv));
    xt::xtensor<double, 1> c = xt::linalg::dot(xt::transpose(A_sqrt), b - a);
    xt::xtensor<double, 2> C_sqrt = xt::linalg::cholesky(C); 
    xt::xtensor<double, 1> c_tilde = xt::linalg::solve(C_sqrt, c); 
    xt::xtensor<double, 2> C_sqrt_inv = xt::linalg::inv(C_sqrt);
    xt::xtensor<double, 2> C_tilde = xt::linalg::dot(C_sqrt_inv, xt::transpose(C_sqrt_inv));

    // Construct matrix M
    xt::xtensor<double, 2> M = xt::zeros<double>({2*nv, 2*nv}); // Initialize M with zeros
    auto tmp_view1 = xt::view(M, xt::range(0, nv), xt::range(0, nv));
    xt::noalias(tmp_view1) = C_tilde;
    auto tmp_view2 = xt::view(M, xt::range(nv, 2*nv), xt::range(nv, 2*nv));
    xt::noalias(tmp_view2) = C_tilde;
    auto tmp_view3 = xt::view(M, xt::range(0, nv), xt::range(nv, 2*nv));
    xt::noalias(tmp_view3) = -xt::eye(nv);
    auto tmp_view4 = xt::view(M, xt::range(nv, 2*nv), xt::range(0, nv));
    xt::noalias(tmp_view4) = xt::linalg::outer(-c_tilde, xt::transpose(c_tilde));

    // Compute the smallest eigenvalue of M
    double lambda_min = xt::amin(xt::real(xt::linalg::eigvals(M)))();

    // Solve for x_rimon
    xt::xtensor<double, 1> x_rimon = xt::linalg::solve(lambda_min * C - xt::eye(nv), xt::linalg::dot(C, c)); 
    x_rimon = a + lambda_min * xt::linalg::solve(xt::transpose(A_sqrt), x_rimon);
    
    return x_rimon;
}

xt::xtensor<double, 1> rimonMethod2d(std::shared_ptr<Ellipsoid2d> SF1, const xt::xtensor<double, 1>& d1, double theta1,
    std::shared_ptr<Ellipsoid2d> SF2, const xt::xtensor<double, 1>& d2, double theta2){

    xt::xtensor<double, 2> A = SF1->getWorldQuadraticCoefficient(theta1);
    xt::xtensor<double, 1> a = SF1->getWorldCenter(d1, theta1);
    xt::xtensor<double, 2> B = SF2->getWorldQuadraticCoefficient(theta2);
    xt::xtensor<double, 1> b = SF2->getWorldCenter(d2, theta2);
    return rimonMethod(A, a, B, b);
}

xt::xtensor<double, 1> rimonMethod3d(std::shared_ptr<Ellipsoid3d> SF1, const xt::xtensor<double, 1>& d1, const xt::xtensor<double, 1>& q1,
    std::shared_ptr<Ellipsoid3d> SF2, const xt::xtensor<double, 1>& d2, const xt::xtensor<double, 1>& q2){

    xt::xtensor<double, 2> A = SF1->getWorldQuadraticCoefficient(q1);
    xt::xtensor<double, 1> a = SF1->getWorldCenter(d1, q1);
    xt::xtensor<double, 2> B = SF2->getWorldQuadraticCoefficient(q2);
    xt::xtensor<double, 1> b = SF2->getWorldCenter(d2, q2);
    return rimonMethod(A, a, B, b);
}