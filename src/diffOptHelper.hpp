#ifndef DIFF_OPT_HELPER_HPP
#define DIFF_OPT_HELPER_HPP

#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xnoalias.hpp>
#include <tuple>
#include <memory>

#include "scalingFunction2d.hpp"
#include "scalingFunction3d.hpp"

/**
 * @brief Get the dual variable in the kkt condition: F1_dp + dual_var * F2_dp = 0
 * 
 * @param F1_dp dF1/dp, shape: (dim_p,)
 * @param F2_dp dF2/dp, shape: (dim_p,)
 * @return double the value of the dual variable at the optimal solution
 */
double getDualVariable(const xt::xtensor<double, 1>& F1_dp, const xt::xtensor<double, 1>& F2_dp);

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
 * @return xt::xtensor<double, 1> dalpha(x)/dx, shape: (dim_x,)
 */
xt::xtensor<double, 1> getGradientGeneral(double dual_var, const xt::xtensor<double, 1>& F1_dp, const xt::xtensor<double, 1>& F2_dp,
    const xt::xtensor<double, 1>& F1_dx, const xt::xtensor<double, 1>& F2_dx,
    const xt::xtensor<double, 2>& F1_dpdp, const xt::xtensor<double, 2>& F2_dpdp,
    const xt::xtensor<double, 2>& F1_dpdx, const xt::xtensor<double, 2>& F2_dpdx);

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
 * @return std::tuple<xt::xtensor<double, 1>,xt::xtensor<double, 2>> dalpha(x)/dx, shape: (dim_x,); d^2alpha(x)/dx^2, shape: (dim_x, dim_x)
 */
std::tuple<xt::xtensor<double, 1>,xt::xtensor<double, 2>> getGradientAndHessianGeneralOld(double dual_var, const xt::xtensor<double, 1>& F1_dp, const xt::xtensor<double, 1>& F2_dp,
    const xt::xtensor<double, 1>& F1_dx, const xt::xtensor<double, 1>& F2_dx,
    const xt::xtensor<double, 2>& F1_dpdp, const xt::xtensor<double, 2>& F2_dpdp,
    const xt::xtensor<double, 2>& F1_dpdx, const xt::xtensor<double, 2>& F2_dpdx,
    const xt::xtensor<double, 2>& F1_dxdx, const xt::xtensor<double, 2>& F2_dxdx,
    const xt::xtensor<double, 3>& F1_dpdpdp, const xt::xtensor<double, 3>& F2_dpdpdp,
    const xt::xtensor<double, 3>& F1_dpdpdx, const xt::xtensor<double, 3>& F2_dpdpdx,
    const xt::xtensor<double, 3>& F1_dpdxdx, const xt::xtensor<double, 3>& F2_dpdxdx);

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
 * @return std::tuple<xt::xtensor<double, 1>,xt::xtensor<double, 2>> dalpha(x)/dx, shape: (dim_x,); d^2alpha(x)/dx^2, shape: (dim_x, dim_x)
 */
std::tuple<xt::xtensor<double, 1>,xt::xtensor<double, 2>> getGradientAndHessianGeneral(double dual_var, const xt::xtensor<double, 1>& F1_dp, const xt::xtensor<double, 1>& F2_dp,
    const xt::xtensor<double, 1>& F1_dx, const xt::xtensor<double, 1>& F2_dx,
    const xt::xtensor<double, 2>& F1_dpdp, const xt::xtensor<double, 2>& F2_dpdp,
    const xt::xtensor<double, 2>& F1_dpdx, const xt::xtensor<double, 2>& F2_dpdx,
    const xt::xtensor<double, 2>& F1_dxdx, const xt::xtensor<double, 2>& F2_dxdx,
    const xt::xtensor<double, 3>& F1_dpdpdp, const xt::xtensor<double, 3>& F2_dpdpdp,
    const xt::xtensor<double, 3>& F1_dpdpdx, const xt::xtensor<double, 3>& F2_dpdpdx,
    const xt::xtensor<double, 3>& F1_dpdxdx, const xt::xtensor<double, 3>& F2_dpdxdx);

/**
 * @brief Get the gradient of alpha^*(x) wrt x in 2D.
 * 
 * @param p Optimal solution p*, shape: (dim_p,)
 * @param SF1 Scaling function 1
 * @param d1 Origin of the body frame of SF1 in the world frame, shape: (dim_p,)
 * @param theta1 Rotation angle representing R(theta1)
 * @param SF2 Scaling function 2
 * @param d2 Origin of the body frame of SF2 in the world frame, shape: (dim_p,)
 * @param theta2 Rotation angle representing R(theta2)
 * @return std::tuple<double, xt::xtensor<double, 1>> [alpha^*(x) is a double, dalpha^*(x)/dx of shape (dim_x,)]
 */
std::tuple<double, xt::xtensor<double, 1>> getGradient2d(
    const xt::xtensor<double, 1>& p, std::shared_ptr<ScalingFunction2d> SF1, const xt::xtensor<double, 1>& d1, double theta1,
    std::shared_ptr<ScalingFunction2d> SF2, const xt::xtensor<double, 1>& d2, double theta2);

/**
 * @brief Get the gradient and Hessian of alpha^*(x) wrt x in 2D.
 * 
 * @param p Optimal solution p*, shape: (dim_p,)
 * @param SF1 Scaling function 1
 * @param d1 Origin of the body frame of SF1 in the world frame, shape: (dim_p,)
 * @param theta1 Rotation angle representing R(theta1)
 * @param SF2 Scaling function 2
 * @param d2 Origin of the body frame of SF2 in the world frame, shape: (dim_p,)
 * @param theta2 Rotation angle representing R(theta2)
 * @return std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> [alpha^*(x) is a double, dalpha^*(x)/dx of shape (dim_x,), 
 * d^2alpha^*(x)/dx^2 of shape (dim_x, dim_x)]
 */
std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> getGradientAndHessian2d(
    const xt::xtensor<double, 1>& p, std::shared_ptr<ScalingFunction2d> SF1, const xt::xtensor<double, 1>& d1, double theta1,
    std::shared_ptr<ScalingFunction2d> SF2, const xt::xtensor<double, 1>& d2, double theta2);

/**
 * @brief Get the gradient of alpha^*(x) wrt x in 3D.
 * 
 * @param p Optimal solution p*, shape: (dim_p,)
 * @param SF1 Scaling function 1
 * @param d1 Origin of the body frame of SF1 in the world frame, shape: (dim_p,)
 * @param q1 Quaternion representing the rotation R(q1)
 * @param SF2 Scaling function 2
 * @param d2 Origin of the body frame of SF2 in the world frame, shape: (dim_p,)
 * @param q2 Quaternion representing the rotation R(q2)
 * @return std::tuple<double, xt::xtensor<double, 1>> [alpha^*(x) is a double, dalpha^*(x)/dx of shape (dim_x,)]
 */
std::tuple<double, xt::xtensor<double, 1>> getGradient3d(
    const xt::xtensor<double, 1>& p, std::shared_ptr<ScalingFunction3d> SF1, const xt::xtensor<double, 1>& d1, const xt::xtensor<double, 1>& q1,
    std::shared_ptr<ScalingFunction3d> SF2, const xt::xtensor<double, 1>& d2, const xt::xtensor<double, 1>& q2);

/**
 * @brief Get the gradient and Hessian of alpha^*(x) wrt x in 3D.
 * 
 * @param p Optimal solution p*, shape: (dim_p,)
 * @param SF1 Scaling function 1
 * @param d1 Origin of the body frame of SF1 in the world frame, shape: (dim_p,)
 * @param q1 Quaternion representing the rotation R(q1)
 * @param SF2 Scaling function 2
 * @param d2 Origin of the body frame of SF2 in the world frame, shape: (dim_p,)
 * @param q2 Quaternion representing the rotation R(q2)
 * @return std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> [alpha^*(x) is a double, dalpha^*(x)/dx of shape (dim_x,), 
 * d^2alpha^*(x)/dx^2 of shape (dim_x, dim_x)]
 */
std::tuple<double, xt::xtensor<double, 1>, xt::xtensor<double, 2>> getGradientAndHessian3d(
    const xt::xtensor<double, 1>& p, std::shared_ptr<ScalingFunction3d> SF1, const xt::xtensor<double, 1>& d1, const xt::xtensor<double, 1>& q1,
    std::shared_ptr<ScalingFunction3d> SF2, const xt::xtensor<double, 1>& d2, const xt::xtensor<double, 1>& q2);

#endif