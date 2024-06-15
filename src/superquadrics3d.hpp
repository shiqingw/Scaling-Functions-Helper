#ifndef SUPERQUADRICS_3D_HPP
#define SUPERQUADRICS_3D_HPP

#include <cmath>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>

#include "scalingFunction3d.hpp"


class Superquadrics3d : public ScalingFunction3d {
    public:
        xt::xarray<double> c; // center of the superquadrics
        xt::xarray<double> a; // positive scalars representing the semi-axes of the superquadrics
        double e1, e2; // positive scalars between 0 and 2

        Superquadrics3d(bool isMoving_, const xt::xarray<double>& c_, const xt::xarray<double>& a_, double e1_, double e2_) : 
            ScalingFunction3d(isMoving_), c(c_), a(a_), e1(e1_), e2(e2_){
            if (c.shape()[0] != 3){
                throw std::invalid_argument("Superquadrics3d: c must be a 3D vector.");
            }
            if (a.shape()[0] != 3){
                throw std::invalid_argument("Superquadrics3d: a must be a 3D vector.");
            }
            if (a(0) <= 0 || a(1) <= 0 || a(2) <= 0){
                throw std::invalid_argument("Superquadrics3d: a must have positive components.");
            }
            if (e1 <= 0 || e1 >= 2 || e2 <= 0 || e2 >= 2){
                throw std::invalid_argument("Superquadrics3d: e1, e2 must be positive scalars between 0 and 2.");
            }
        }; 
        ~Superquadrics3d() = default;

        /**
         * @brief Calculate the scaling function F(P) = [|x-c1/a1|^(2/e2)+|y-c2/a2|^(2/e2)]^(e2/e1)+|z-c3/a3|^(2/e1).
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return double F(P)
         */
        double getBodyF(const xt::xarray<double>& P) const override;

        /**
         * @brief Calculate the gradient of the scaling function 
         * F(P) = [|x-c1/a1|^(2/e2)+|y-c2/a2|^(2/e2)]^(e2/e1)+|z-c3/a3|^(2/e1) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> dF/dP, shape: (dim_p,)
         */
        xt::xarray<double> getBodyFdP(const xt::xarray<double>& P) const override;

        /**
         * @brief Calculate the Hessian of the scaling function 
         * F(P) = [|x-c1/a1|^(2/e2)+|y-c2/a2|^(2/e2)]^(e2/e1)+|z-c3/a3|^(2/e1) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> d^2F/dPdP, shape: (dim_p, dim_p)
         */
        xt::xarray<double> getBodyFdPdP(const xt::xarray<double>& P) const override;

        /**
         * @brief Calculate the third order derivative of the scaling function 
         * F(P) = [|x-c1/a1|^(2/e2)+|y-c2/a2|^(2/e2)]^(e2/e1)+|z-c3/a3|^(2/e1) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> d^3F/dPdPdP, shape: (dim_p, dim_p, dim_p)
         */
        xt::xarray<double> getBodyFdPdPdP(const xt::xarray<double>& P) const override;
}; 


#endif // SUPERQUADRICS_3D_HPP
