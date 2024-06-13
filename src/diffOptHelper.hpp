#ifndef DIFF_OPT_HELPER_HPP
#define DIFF_OPT_HELPER_HPP

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <tuple>

/**
 * @brief Get the dual variable in the kkt condition: F1_dp + dual_var * F2_dp = 0
 * 
 * @param F1_dp dF1/dp, shape: (dim_p,)
 * @param F2_dp dF2/dp, shape: (dim_p,)
 * @return double the value of the dual variable at the optimal solution
 */
double getDualVariable(const xt::xarray<double>& F1_dp, const xt::xarray<double>& F2_dp);

/**
 * @brief Get the gradient of alpha(x) = F1(p,x) wrt x.
 * 
 * @param dual_var Value of the dual variable at the optimal solution
 * @param F1_dp dF1/dp, shape: (dim_p,)
 * @param F2_dp dF2/dp, shape: (dim_p,)
 * @param F1_dx dF1/dx, shape: (dim_x,)
 * @param F2_dx dF2/dx, shape: (dim_x,)
 * @param F1_dpdp d^2F1/dp^2, shape: (dim_p, dim_p)
 * @param F2_dpdp d^2F2/dp^2, shape: (dim_p, dim_p)
 * @param F1_dpdx d^2F1/dpdx, shape: (dim_p, dim_x)
 * @param F2_dpdx d^2F2/dpdx, shape: (dim_p, dim_x)
 * @return xt::xarray<double> dalpha(x)/dx, shape: (dim_x,)
 */
xt::xarray<double> getGradientGeneral(double dual_var, const xt::xarray<double>& F1_dp, const xt::xarray<double>& F2_dp,
    const xt::xarray<double>& F1_dx, const xt::xarray<double>& F2_dx,
    const xt::xarray<double>& F1_dpdp, const xt::xarray<double>& F2_dpdp,
    const xt::xarray<double>& F1_dpdx, const xt::xarray<double>& F2_dpdx);

/**
 * @brief Get the gradient and hessian of alpha(x) = F1(p,x) wrt x.
 * 
 * @param dual_var Value of the dual variable at the optimal solution
 * @param F1_dp dF1/dp, shape: (dim_p,)
 * @param F2_dp dF2/dp, shape: (dim_p,)
 * @param F1_dx dF1/dx, shape: (dim_x,)
 * @param F2_dx dF2/dx, shape: (dim_x,)
 * @param F1_dpdp d^2F1/dp^2, shape: (dim_p, dim_p)
 * @param F2_dpdp d^2F2/dp^2, shape: (dim_p, dim_p)
 * @param F1_dpdx d^2F1/dpdx, shape: (dim_p, dim_x)
 * @param F2_dpdx d^2F2/dpdx, shape: (dim_p, dim_x)
 * @param F1_dxdx d^2F1/dx^2, shape: (dim_x, dim_x)
 * @param F2_dxdx d^2F2/dx^2, shape: (dim_x, dim_x)
 * @param F1_dpdpdp d^3F1/dp^3, shape: (dim_p, dim_p, dim_p)
 * @param F2_dpdpdp d^3F2/dp^3, shape: (dim_p, dim_p, dim_p)
 * @param F1_dpdpdx d^3F1/dp^2dx, shape: (dim_p, dim_p, dim_x)
 * @param F2_dpdpdx d^3F2/dp^2dx, shape: (dim_p, dim_p, dim_x)
 * @param F1_dpdxdx d^3F1/dpdx^2, shape: (dim_p, dim_x, dim_x)
 * @param F2_dpdxdx d^3F2/dpdx^2, shape: (dim_p, dim_x, dim_x)
 * @return std::tuple<xt::xarray<double>,xt::xarray<double>> dalpha(x)/dx, shape: (dim_x,); d^2alpha(x)/dx^2, shape: (dim_x, dim_x)
 */
std::tuple<xt::xarray<double>,xt::xarray<double>> getGradientAndHessianGeneral(double dual_var, const xt::xarray<double>& F1_dp, const xt::xarray<double>& F2_dp,
    const xt::xarray<double>& F1_dx, const xt::xarray<double>& F2_dx,
    const xt::xarray<double>& F1_dpdp, const xt::xarray<double>& F2_dpdp,
    const xt::xarray<double>& F1_dpdx, const xt::xarray<double>& F2_dpdx,
    const xt::xarray<double>& F1_dxdx, const xt::xarray<double>& F2_dxdx,
    const xt::xarray<double>& F1_dpdpdp, const xt::xarray<double>& F2_dpdpdp,
    const xt::xarray<double>& F1_dpdpdx, const xt::xarray<double>& F2_dpdpdx,
    const xt::xarray<double>& F1_dpdxdx, const xt::xarray<double>& F2_dpdxdx);

#endif //DIFF_OPT_HELPER_HPP