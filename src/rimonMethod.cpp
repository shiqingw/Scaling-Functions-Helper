#include "rimonMethod.hpp"

xt::xarray<double> rimonMethod(const xt::xarray<double>& A, const xt::xarray<double>& a, const xt::xarray<double>& B, const xt::xarray<double>& b) {
    int nv = A.shape()[0];

    xt::xarray<double> A_sqrt = xt::linalg::cholesky(A); 
    xt::xarray<double> A_sqrt_inv = xt::linalg::inv(A_sqrt);
    xt::xarray<double> C = xt::linalg::dot(xt::linalg::dot(A_sqrt_inv, B), xt::transpose(A_sqrt_inv));
    xt::xarray<double> c = xt::linalg::dot(xt::transpose(A_sqrt), b - a);
    xt::xarray<double> C_sqrt = xt::linalg::cholesky(C); 
    xt::xarray<double> c_tilde = xt::linalg::solve(C_sqrt, c); 
    xt::xarray<double> C_sqrt_inv = xt::linalg::inv(C_sqrt);
    xt::xarray<double> C_tilde = xt::linalg::dot(C_sqrt_inv, xt::transpose(C_sqrt_inv));

    // Construct matrix M
    xt::xarray<double> M = xt::zeros<double>({2*nv, 2*nv}); // Initialize M with zeros
    xt::view(M, xt::range(0, nv), xt::range(0, nv)) = C_tilde;
    xt::view(M, xt::range(nv, 2*nv), xt::range(nv, 2*nv)) = C_tilde;
    xt::view(M, xt::range(0, nv), xt::range(nv, 2*nv)) = -xt::eye(nv);
    xt::view(M, xt::range(nv, 2*nv), xt::range(0, nv)) = xt::linalg::outer(-c_tilde, xt::transpose(c_tilde));

    // Compute the smallest eigenvalue of M
    double lambda_min = xt::amin(xt::real(xt::linalg::eigvals(M)))();

    // Solve for x_rimon
    xt::xarray<double> x_rimon = xt::linalg::solve(lambda_min * C - xt::eye(nv), xt::linalg::dot(C, c)); 
    x_rimon = a + lambda_min * xt::linalg::solve(xt::transpose(A_sqrt), x_rimon);
    
    return x_rimon;
}