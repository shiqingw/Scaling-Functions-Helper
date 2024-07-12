#ifndef HYPERPLANE_2D_HPP
#define HYPERPLANE_2D_HPP

#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "scalingFunction2d.hpp"


class Hyperplane2d : public ScalingFunction2d {
    public:
        xt::xtensor<double, 1> a; // vector a in F(P) = a^T P + b, shape: (2,)
        double b; // scalar b in F(P) = a^T P + b

        /**
         * @brief Construct a new Hyperplane2d object
         * 
         * @param isMoving_ True if the scaling function is moving, false otherwise
         * @param a_ Vector a in F(P) = a^T P + b, shape: (2,)
         * @param b_ Scalar b in F(P) = a^T P + b
         */
        Hyperplane2d(bool isMoving_, const xt::xtensor<double, 1>& a_, double b_) 
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
        double getBodyF(const xt::xtensor<double, 1>& P) const override;

        /**
         * @brief Calculate the gradient of the scaling function F(P) = a^T P + b w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 1> dF/dP, shape: (dim_p,)
         */
        xt::xtensor<double, 1> getBodyFdP(const xt::xtensor<double, 1>& P) const override;

        /**
         * @brief Calculate the Hessian of the scaling function F(P) = a^T P + b w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 2> d^2F/dPdP, shape: (dim_p, dim_p)
         */
        xt::xtensor<double, 2> getBodyFdPdP(const xt::xtensor<double, 1>& P) const override;

        /**
         * @brief Calculate the third order derivative of the scaling function F(P) = a^T P + b w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 3> d^3F/dPdPdP, shape: (dim_p, dim_p, dim_p)
         */
        xt::xtensor<double, 3> getBodyFdPdPdP(const xt::xtensor<double, 1>& P) const override;
}; 


#endif // HYPERPLANE_2D_HPP
