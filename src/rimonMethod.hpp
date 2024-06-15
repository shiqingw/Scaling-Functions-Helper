#ifndef RIMON_METHOD_HPP
#define RIMON_METHOD_HPP

#include <cmath>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
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

#endif // RIMON_METHOD_HPP