#ifndef ELLIPSOID_2D_HPP
#define ELLIPSOID_2D_HPP

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>

#include "scalingFunction2d.hpp"


class Ellipsoid2d : public ScalingFunction2d {
    public:
        xt::xarray<double> Q; // Symmetric quadratic coefficient, shape: (2, 2)
        xt::xarray<double> mu; // Center of the ellipsoid, shape: (2,)

        Ellipsoid2d(bool isMoving_, const xt::xarray<double>& Q_, const xt::xarray<double>& mu_) 
        : ScalingFunction2d(isMoving_), Q(Q_), mu(mu_) {
            // if Q_ is not symmetric, return an error
            if (!xt::isclose(Q_, xt::transpose(Q_, {1, 0}))()){
                throw std::invalid_argument("Q must be symmetric.");
            }
            if (Q_.shape()[0] != 2 || Q_.shape()[1] != 2){
                throw std::invalid_argument("Q must be of shape (2,2).");
            }
            if (mu_.shape()[0] != 2){
                throw std::invalid_argument("mu must be of shape (2,).");
            }
        };
        ~Ellipsoid2d() = default;

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

         /**
         * @brief Get the quadratic coefficient Q' = R Q R^T in the world frame.
         * 
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> Q', shape: (dim_p, dim_p)
         */
        xt::xarray<double> getWorldQuadraticCoefficient(double theta) const;

        /**
         * @brief Get the center of the ellipsoid mu' = d + R mu in the world frame.
         * 
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> mu', shape: (dim_p,)
         */
        xt::xarray<double> getWorldCenter(const xt::xarray<double>& d, double theta) const;
}; 


#endif // ELLIPSOID_2D_HPP
