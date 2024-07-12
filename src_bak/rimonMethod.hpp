#ifndef RIMON_METHOD_HPP
#define RIMON_METHOD_HPP

#include <cmath>
#include <memory>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>

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
 * @return xt::xarray<double> p_rimon, shape=(dim_p,)
 */
xt::xarray<double> rimonMethod(const xt::xarray<double>& A, const xt::xarray<double>& a, 
                                const xt::xarray<double>& B, const xt::xarray<double>& b);
/**
 * @brief Apply the Rimon method to two 2D ellipsoids.
 * 
 * @param SF1 Scaling function 1
 * @param d1 Origin of the body frame of SF1 in the world frame, shape: (dim_p,)
 * @param theta1 Rotation angle representing R(theta1)
 * @param SF2 Scaling function 2
 * @param d2 Origin of the body frame of SF2 in the world frame, shape: (dim_p,)
 * @param theta2 Rotation angle representing R(theta2)
 * @return xt::xarray<double> p_rimon, shape=(dim_p,)
 */
xt::xarray<double> rimonMethod2d(std::shared_ptr<Ellipsoid2d> SF1, const xt::xarray<double>& d1, double theta1,
    std::shared_ptr<Ellipsoid2d> SF2, const xt::xarray<double>& d2, double theta2);

/**
 * @brief Apply the Rimon method to two 3D ellipsoids.
 * 
 * @param SF1 Scaling function 1
 * @param d1 Origin of the body frame of SF1 in the world frame, shape: (dim_p,)
 * @param q1 Quaternion representing the rotation R(q1)
 * @param SF2 Scaling function 2
 * @param d2 Origin of the body frame of SF2 in the world frame, shape: (dim_p,)
 * @param q2 Quaternion representing the rotation R(q2)
 * @return xt::xarray<double> p_rimon, shape=(dim_p,)
 */
xt::xarray<double> rimonMethod3d(std::shared_ptr<Ellipsoid3d> SF1, const xt::xarray<double>& d1, const xt::xarray<double>& q1,
    std::shared_ptr<Ellipsoid3d> SF2, const xt::xarray<double>& d2, const xt::xarray<double>& q2);

#endif // RIMON_METHOD_HPP