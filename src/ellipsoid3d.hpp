#ifndef ELLIPSOID_3D_HPP
#define ELLIPSOID_3D_HPP

#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "scalingFunction3d.hpp"


class Ellipsoid3d : public ScalingFunction3d {
    public:
        xt::xtensor<double, 2> Q; // Symmetric quadratic coefficient, shape: (3, 3)
        xt::xtensor<double, 1> mu; // Center of the ellipsoid, shape: (3,)

        /**
         * @brief Construct a new Ellipsoid3d object.
         * 
         * @param isMoving_ Whether the scaling function is moving with the body frame
         * @param Q_ Symmetric quadratic coefficient, shape: (3, 3)
         * @param mu_ Center of the ellipsoid, shape: (3,)
         */
        Ellipsoid3d(bool isMoving_, const xt::xtensor<double, 2>& Q_, const xt::xtensor<double, 1>& mu_) 
        : ScalingFunction3d(isMoving_), Q(Q_), mu(mu_) {
            // if Q_ is not symmetric, return an error
            if (!xt::isclose(Q_, xt::transpose(Q_, {1, 0}))()){
                throw std::invalid_argument("Q must be symmetric.");
            }
            if (Q_.shape()[0] != 3 || Q_.shape()[1] != 3){
                throw std::invalid_argument("Q must be of shape (3,3).");
            }
            if (mu_.shape()[0] != 3){
                throw std::invalid_argument("mu must be of shape (3,).");
            }
        };
        ~Ellipsoid3d() = default;

        /**
         * @brief Calculate the scaling function F(P) = (P-mu)^T Q (P-mu).
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return double F(P)
         */
        double getBodyF(const xt::xtensor<double, 1>& P) const override;

        /**
         * @brief Calculate the gradient of the scaling function F(P) = (P-mu)^T Q (P-mu) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 1> dF/dP, shape: (dim_p,)
         */
        xt::xtensor<double, 1> getBodyFdP(const xt::xtensor<double, 1>& P) const override;

        /**
         * @brief Calculate the Hessian of the scaling function F(P) = (P-mu)^T Q (P-mu) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 2> d^2F/dPdP, shape: (dim_p, dim_p)
         */
        xt::xtensor<double, 2> getBodyFdPdP(const xt::xtensor<double, 1>& P) const override;

        /**
         * @brief Calculate the third order derivative of the scaling function F(P) = (P-mu)^T Q (P-mu) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 3> d^3F/dPdPdP, shape: (dim_p, dim_p, dim_p)
         */
        xt::xtensor<double, 3> getBodyFdPdPdP(const xt::xtensor<double, 1>& P) const override;

        /**
         * @brief Get the quadratic coefficient Q' = R Q R^T in the world frame.
         * 
         * @param q Unit quaternion [qx,qy,qz,qw] representing R(q), shape: (dim_q,)
         * @return xt::xtensor<double, 2> Q', shape: (dim_p, dim_p)
         */
        xt::xtensor<double, 2> getWorldQuadraticCoefficient(const xt::xtensor<double, 1>& q) const;

        /**
         * @brief Get the center of the ellipsoid mu' = d + R mu in the world frame.
         * 
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param q Unit quaternion [qx,qy,qz,qw] representing R(q), shape: (dim_q,)
         * @return xt::xtensor<double, 1> mu', shape: (dim_p,)
         */
        xt::xtensor<double, 1> getWorldCenter(const xt::xtensor<double, 1>& d, const xt::xtensor<double, 1>& q) const;

}; 


#endif // ELLIPSOID_3D_HPP
