#include "smoothMinimum.hpp"

std::tuple<double, xt::xarray<double>, xt::xarray<double>>
getSmoothMinimumAndLocalGradientAndHessian(const double rho, const xt::xarray<double>& h) {
    
    int dim_h = h.shape()[0];
    double c = xt::amin(h)();
    xt::xarray<double> z = xt::exp(-rho*(h - c))/(double)dim_h;
    double sum_z = xt::sum(z)();
    double F = -log(sum_z)/rho + c;

    xt::xarray<double> F_dh = z / sum_z; // shape N

    xt::xarray<double> diag_z = xt::diag(z);
    xt::xarray<double> z_zT = xt::linalg::outer(z, z);
    xt::xarray<double> F_dhdh = -rho * (diag_z/sum_z - z_zT/pow(sum_z,2));

    return std::make_tuple(F, F_dh, F_dhdh);
}

std::tuple<double, xt::xarray<double>, xt::xarray<double>> 
getSmoothMinimumAndTotalGradientAndHessian(const double rho, const xt::xarray<double>& h,
const xt::xarray<double>& h_dx, const xt::xarray<double>& h_dxdx){

    double F;
    xt::xarray<double> F_dh, F_dhdh;
    std::tie(F, F_dh, F_dhdh) = getSmoothMinimumAndLocalGradientAndHessian(rho, h);

    xt::xarray<double> F_dx = xt::linalg::dot(F_dh, h_dx); // shape dim_x

    xt::xarray<double> F_dxdx = xt::linalg::dot(xt::transpose(h_dx), xt::linalg::dot(F_dhdh, h_dx));
    F_dxdx += xt::linalg::tensordot(F_dh, h_dxdx, {0}, {0});

    return std::make_tuple(F, F_dx, F_dxdx);
}