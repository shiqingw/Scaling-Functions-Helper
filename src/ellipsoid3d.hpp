#ifndef ELLIPSOID_3D_HPP
#define ELLIPSOID_3D_HPP

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>

#include "scalingFunction3d.hpp"


class Ellipsoid3d : public ScalingFunction3d {
    public:
        xt::xarray<double> Q; // Symmetric quadratic coefficient, shape: (3, 3)
        xt::xarray<double> mu; // Center of the ellipsoid, shape: (3,)

        Ellipsoid3d(bool isMoving_, const xt::xarray<double>& Q_, const xt::xarray<double>& mu_) 
        : ScalingFunction3d(isMoving_), Q(Q_), mu(mu_) {
            // if Q_ is not symmetric, return an error
            if (!xt::isclose(Q_, xt::transpose(Q_, {1, 0}))()){
                throw std::invalid_argument("Q must be symmetric.");
            }
        };
        ~Ellipsoid3d() = default;

        /**
         * @brief Calculate the scaling function F(P) = (P-mu)^T Q (P-mu).
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return double F(P)
         */
        double getBodyF(const xt::xarray<double>& P) const override;

        /**
         * @brief Calculate the gradient of the scaling function F(P) = (P-mu)^T Q (P-mu) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> dF/dP, shape: (dim_p,)
         */
        xt::xarray<double> getBodyFdP(const xt::xarray<double>& P) const override;

        /**
         * @brief Calculate the Hessian of the scaling function F(P) = (P-mu)^T Q (P-mu) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> d^2F/dPdP, shape: (dim_p, dim_p)
         */
        xt::xarray<double> getBodyFdPdP(const xt::xarray<double>& P) const override;

        /**
         * @brief Calculate the third order derivative of the scaling function F(P) = (P-mu)^T Q (P-mu) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> d^3F/dPdPdP, shape: (dim_p, dim_p, dim_p)
         */
        xt::xarray<double> getBodyFdPdPdP(const xt::xarray<double>& P) const override;
}; 


#endif // ELLIPSOID_3D_HPP
