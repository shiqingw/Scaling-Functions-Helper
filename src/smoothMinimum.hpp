#ifndef SMOOTH_MINIMUM_HPP
#define SMOOTH_MINIMUM_HPP

#include <cmath>
#include <tuple>
#include <iostream>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

/**
 * @brief Get the local derivatives of the smooth minimum function F(h) = -1/rho*log[sum(exp[-rho*h_i])/len(h)] wrt h.
 * 
 * @param rho A strictly positive tuning parameter
 * @param h A real vector, shape: (dim_h,)
 * @return std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> F(h), dF/dh, d^2F/dh^2
 */
std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>>
getSmoothMinimumAndLocalGradientAndHessian(const double rho, const xt::xtensor<double, 1>& h);

/**
 * @brief Get the gradient and Hessian of the smooth minimum function F(h(x)) = -1/rho*log[sum(exp[-rho*h_i])/len(h)] wrt x.
 * 
 * @param rho A strictly positive scalar
 * @param h A real vector, shape: (dim_h,)
 * @param h_dx dh/dx, shape: (dim_h, dim_x)
 * @param h_dxdx d^2h/dxdx, shape: (dim_h, dim_x, dim_x)
 * @return std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> F(h), dF/dx, d^2F/dxdx
 */
std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> 
getSmoothMinimumAndTotalGradientAndHessian(const double rho, const xt::xtensor<double, 1>& h,
const xt::xtensor<double, 2>& h_dx, const xt::xtensor<double, 3>& h_dxdx);

#endif // SMOOTH_MINIMUM_HPP