#ifndef SUPERQUADRICS_3D_HPP
#define SUPERQUADRICS_3D_HPP

#include <cmath>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "scalingFunction3d.hpp"


class Superquadrics3d : public ScalingFunction3d {
    public:
        xt::xtensor<double, 1> c; // center of the superquadrics
        xt::xtensor<double, 1> a; // positive scalars representing the semi-axes of the superquadrics
        double e1, e2; // positive scalars between 0 and 2

        /**
         * @brief Construct a new Superquadrics3d object
         * 
         * @param isMoving_ True if the scaling function is moving, false otherwise
         * @param c_ Center of the superquadrics, shape: (3,)
         * @param a_ Positive scalars representing the semi-axes of the superquadrics, shape: (3,)
         * @param e1_ Positive scalar between 0 and 2
         * @param e2_ Positive scalar between 0 and 2
         */
        Superquadrics3d(bool isMoving_, const xt::xtensor<double, 1>& c_, const xt::xtensor<double, 1>& a_, double e1_, double e2_) : 
            ScalingFunction3d(isMoving_), c(c_), a(a_), e1(e1_), e2(e2_){
            if (c.shape()[0] != 3){
                throw std::invalid_argument("Superquadrics3d: c must be a vector of length 3.");
            }
            if (a.shape()[0] != 3){
                throw std::invalid_argument("Superquadrics3d: a must be a vector of length 3.");
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
        double getBodyF(const xt::xtensor<double, 1>& P) const override;

        /**
         * @brief Calculate the gradient of the scaling function 
         * F(P) = [|x-c1/a1|^(2/e2)+|y-c2/a2|^(2/e2)]^(e2/e1)+|z-c3/a3|^(2/e1) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 1> dF/dP, shape: (dim_p,)
         */
        xt::xtensor<double, 1> getBodyFdP(const xt::xtensor<double, 1>& P) const override;

        /**
         * @brief Calculate the Hessian of the scaling function 
         * F(P) = [|x-c1/a1|^(2/e2)+|y-c2/a2|^(2/e2)]^(e2/e1)+|z-c3/a3|^(2/e1) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 2> d^2F/dPdP, shape: (dim_p, dim_p)
         */
        xt::xtensor<double, 2> getBodyFdPdP(const xt::xtensor<double, 1>& P) const override;

        /**
         * @brief Calculate the third order derivative of the scaling function 
         * F(P) = [|x-c1/a1|^(2/e2)+|y-c2/a2|^(2/e2)]^(e2/e1)+|z-c3/a3|^(2/e1) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 3> d^3F/dPdPdP, shape: (dim_p, dim_p, dim_p)
         */
        xt::xtensor<double, 3> getBodyFdPdPdP(const xt::xtensor<double, 1>& P) const override;
}; 


#endif // SUPERQUADRICS_3D_HPP
