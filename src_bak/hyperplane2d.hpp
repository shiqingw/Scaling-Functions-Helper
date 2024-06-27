#ifndef HYPERPLANE_2D_HPP
#define HYPERPLANE_2D_HPP

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>

#include "scalingFunction2d.hpp"


class Hyperplane2d : public ScalingFunction2d {
    public:
        xt::xarray<double> a; // vector a in F(P) = a^T P + b, shape: (2,)
        double b; // scalar b in F(P) = a^T P + b

        Hyperplane2d(bool isMoving_, const xt::xarray<double>& a_, double b_) 
        : ScalingFunction2d(isMoving_), a(a_), b(b_) {
            if (a.shape()[0] != 2){
                throw std::invalid_argument("a must have shape (3,)");
            }
        };
        ~Hyperplane2d() = default;

        /**
         * @brief Calculate the scaling function F(P) = a^T P + b.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return double F(P)
         */
        double getBodyF(const xt::xarray<double>& P) const override;

        /**
         * @brief Calculate the gradient of the scaling function F(P) = a^T P + b w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> dF/dP, shape: (dim_p,)
         */
        xt::xarray<double> getBodyFdP(const xt::xarray<double>& P) const override;

        /**
         * @brief Calculate the Hessian of the scaling function F(P) = a^T P + b w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> d^2F/dPdP, shape: (dim_p, dim_p)
         */
        xt::xarray<double> getBodyFdPdP(const xt::xarray<double>& P) const override;

        /**
         * @brief Calculate the third order derivative of the scaling function F(P) = a^T P + b w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> d^3F/dPdPdP, shape: (dim_p, dim_p, dim_p)
         */
        xt::xarray<double> getBodyFdPdPdP(const xt::xarray<double>& P) const override;
}; 


#endif // HYPERPLANE_2D_HPP
