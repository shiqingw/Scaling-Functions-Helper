#ifndef LOG_SUM_EXP_2D_HPP
#define LOG_SUM_EXP_2D_HPP

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>

#include "scalingFunction2d.hpp"

class LogSumExp2d : public ScalingFunction2d {
    public:
        xt::xtensor<double, 2> A; // Shape: (N, 2)
        xt::xtensor<double, 1> b; // Shape: (N,)
        double kappa; // positive scalar

        /**
         * @brief Construct a new LogSumExp2d object
         * 
         * @param isMoving_ True if the scaling function is moving, false otherwise
         * @param A_ Matrix A in F(P) = log[sum(exp[k(A P + b)])/len(A)] + 1, shape: (N, 2)
         * @param b_ Vector b in F(P) = log[sum(exp[k(A P + b)])/len(A)] + 1, shape: (N,)
         * @param kappa_ Positive scalar k in F(P) = log[sum(exp[k(A P + b)])/len(A)] + 1
         */
        LogSumExp2d(bool isMoving_, const xt::xtensor<double, 2>& A_, const xt::xtensor<double, 1>& b_,
                    double kappa_):ScalingFunction2d(isMoving_), A(A_), b(b_), kappa(kappa_) {
            if (A_.shape()[1] != 2){
                throw std::invalid_argument("A must have 2 columns.");
            }
            if (A_.shape()[0] != b_.shape()[0]){
                throw std::invalid_argument("A and b must have the same number of rows.");
            }
            if (kappa_ <= 0){
                throw std::invalid_argument("kappa must be positive.");
            }
            
        }
        ~LogSumExp2d() = default;

        /**
         * @brief Calculate the scaling function F(P) = log[sum(exp[k(A P + b)])/len(A)] + 1.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return double F(P)
         */
        double getBodyF(const xt::xtensor<double, 1>& P) const override;

        /**
         * @brief Calculate the gradient of the scaling function F(P) = log[sum(exp[k(A P + b)])/len(A)] + 1 
         * w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 1> dF/dP, shape: (dim_p,)
         */
        xt::xtensor<double, 1> getBodyFdP(const xt::xtensor<double, 1>& P) const override;

        /**
         * @brief Calculate the Hessian of the scaling function F(P) = log[sum(exp[k(A P + b)])/len(A)] + 1 
         * w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 2> d^2F/dPdP, shape: (dim_p, dim_p)
         */
        xt::xtensor<double, 2> getBodyFdPdP(const xt::xtensor<double, 1>& P) const override;

        /**
         * @brief Calculate the third order derivative of the scaling function 
         * F(P) = log[sum(exp[k(A P + b)])/len(A)] + 1 w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 3> d^3F/dPdPdP, shape: (dim_p, dim_p, dim_p)
         */
        xt::xtensor<double, 3> getBodyFdPdPdP(const xt::xtensor<double, 1>& P) const override;

        /**
         * @brief Get the matrix coefficient A' = A R^T in the world frame.
         * 
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 2> A', shape: (N, dim_p)
         */
        xt::xtensor<double, 2> getWorldMatrixCoefficient(double theta) const;

        /**
         * @brief Get the vector coefficient b' = - A R^T d + b in the world frame.
         * 
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 1> b', shape: (N,)
         */
        xt::xtensor<double, 1> getWorldVectorCoefficient(const xt::xtensor<double, 1>& d, double theta) const;
}; 


#endif // LOG_SUM_EXP_3D_HPP
