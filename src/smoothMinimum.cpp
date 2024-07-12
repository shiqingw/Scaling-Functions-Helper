#include "smoothMinimum.hpp"

std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>>
getSmoothMinimumAndLocalGradientAndHessian(const double rho, const xt::xtensor<double, 1>& h) {
    
    int dim_h = h.shape()[0];
    double c = xt::amin(h)();
    xt::xtensor<double, 1> z = xt::exp(-rho*(h - c))/(double)dim_h;
    double sum_z = xt::sum(z)();
    double F = -log(sum_z)/rho + c;

    xt::xtensor<double, 1> F_dh = z / sum_z; // shape (N, )

    xt::xtensor<double, 2> diag_z = xt::diag(z); // shape (N, N)
    xt::xtensor<double, 2> z_zT = xt::linalg::outer(z, z); // shape (N, N)
    xt::xtensor<double, 2> F_dhdh = -rho * (diag_z/sum_z - z_zT/pow(sum_z,2)); // shape (N, N)

    return std::make_tuple(F, F_dh, F_dhdh);
}

std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> 
getSmoothMinimumAndTotalGradientAndHessian(const double rho, const xt::xtensor<double, 1>& h,
const xt::xtensor<double, 2>& h_dx, const xt::xtensor<double, 3>& h_dxdx){

    double F;
    xt::xtensor<double, 1> F_dh;
    xt::xtensor<double, 2> F_dhdh;
    std::tie(F, F_dh, F_dhdh) = getSmoothMinimumAndLocalGradientAndHessian(rho, h);

    xt::xtensor<double, 1> F_dx = xt::linalg::dot(F_dh, h_dx); // shape dim_x

    xt::xtensor<double, 2> F_dxdx = xt::linalg::dot(xt::transpose(h_dx), xt::linalg::dot(F_dhdh, h_dx));
    F_dxdx += xt::linalg::tensordot(F_dh, h_dxdx, {0}, {0});

    return std::make_tuple(F, F_dx, F_dxdx);
}