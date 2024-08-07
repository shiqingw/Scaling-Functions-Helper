#ifndef RIMON_METHOD_HPP
#define RIMON_METHOD_HPP

#include <cmath>
#include <memory>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xnoalias.hpp>

#include "ellipsoid2d.hpp"
#include "ellipsoid3d.hpp"

/**
 * @brief Consider the ellipses(2D)/ellipsoids(3D) F_A(p) = (p-a)^T A (p-a) <=1 and F_B(p) = (p-b)^T B (p-b) <=1.
 * The Rimon method computes the point p_rimon where the elliptical surface F_A first touches F_B.
 * 
 * @param A Quadratic matrix A=A^T>0 of the first ellipse/ellipsoid, shape=(dim_p,dim_p)
 * @param a Center of the first ellipse/ellipsoid, shape=(dim_p,)
 * @param B Quadratic matrix B=B^T>0 of the second ellipse/ellipsoid, shape=(dim_p,dim_p)
 * @param b Center of the second ellipse/ellipsoid, shape=(dim_p,)
 * @return xt::xtensor<double, 1> p_rimon, shape=(dim_p,)
 */
xt::xtensor<double, 1> rimonMethod(const xt::xtensor<double, 2>& A, const xt::xtensor<double, 1>& a, 
                                const xt::xtensor<double, 2>& B, const xt::xtensor<double, 1>& b);
/**
 * @brief Apply the Rimon method to two 2D ellipsoids.
 * 
 * @param SF1 Scaling function 1
 * @param d1 Origin of the body frame of SF1 in the world frame, shape: (dim_p,)
 * @param theta1 Rotation angle representing R(theta1)
 * @param SF2 Scaling function 2
 * @param d2 Origin of the body frame of SF2 in the world frame, shape: (dim_p,)
 * @param theta2 Rotation angle representing R(theta2)
 * @return xt::xtensor<double, 1> p_rimon, shape=(dim_p,)
 */
xt::xtensor<double, 1> rimonMethod2d(std::shared_ptr<Ellipsoid2d> SF1, const xt::xtensor<double, 1>& d1, double theta1,
    std::shared_ptr<Ellipsoid2d> SF2, const xt::xtensor<double, 1>& d2, double theta2);

/**
 * @brief Apply the Rimon method to two 3D ellipsoids.
 * 
 * @param SF1 Scaling function 1
 * @param d1 Origin of the body frame of SF1 in the world frame, shape: (dim_p,)
 * @param q1 Quaternion representing the rotation R(q1)
 * @param SF2 Scaling function 2
 * @param d2 Origin of the body frame of SF2 in the world frame, shape: (dim_p,)
 * @param q2 Quaternion representing the rotation R(q2)
 * @return xt::xtensor<double, 1> p_rimon, shape=(dim_p,)
 */
xt::xtensor<double, 1> rimonMethod3d(std::shared_ptr<Ellipsoid3d> SF1, const xt::xtensor<double, 1>& d1, const xt::xtensor<double, 1>& q1,
    std::shared_ptr<Ellipsoid3d> SF2, const xt::xtensor<double, 1>& d2, const xt::xtensor<double, 1>& q2);

#endif // RIMON_METHOD_HPP