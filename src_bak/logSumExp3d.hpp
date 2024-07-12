#ifndef LOG_SUM_EXP_3D_HPP
#define LOG_SUM_EXP_3D_HPP

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>

#include "scalingFunction3d.hpp"


class LogSumExp3d : public ScalingFunction3d {
    public:
        xt::xarray<double> A; // Shape: (N, 3)
        xt::xarray<double> b; // Shape: (N,)
        double kappa; // positive scalar

        /**
         * @brief Construct a new LogSumExp3d object
         * 
         * @param isMoving_ True if the scaling function is moving, false otherwise
         * @param A_ Matrix A in F(P) = log[sum(exp[k(A P + b)])/len(A)] + 1, shape: (N, 3)
         * @param b_ Vector b in F(P) = log[sum(exp[k(A P + b)])/len(A)] + 1, shape: (N,)
         * @param kappa_ Positive scalar k in F(P) = log[sum(exp[k(A P + b)])/len(A)] + 1
         */
        LogSumExp3d(bool isMoving_, const xt::xarray<double>& A_, const xt::xarray<double>& b_,
                    double kappa_):ScalingFunction3d(isMoving_), A(A_), b(b_), kappa(kappa_) {
            if (A_.shape()[1] != 3){
                throw std::invalid_argument("A must have 3 columns.");
            }
            if (A_.shape()[0] != b_.shape()[0]){
                throw std::invalid_argument("A and b must have the same number of rows.");
            }
            if (kappa_ <= 0){
                throw std::invalid_argument("kappa must be positive.");
            }
            
        }
        ~LogSumExp3d() = default;

        /**
         * @brief Calculate the scaling function F(P) = log[sum(exp[k(A P + b)])/len(A)] + 1.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return double F(P)
         */
        double getBodyF(const xt::xarray<double>& P) const override;

        /**
         * @brief Calculate the gradient of the scaling function F(P) = log[sum(exp[k(A P + b)])/len(A)] + 1 
         * w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> dF/dP, shape: (dim_p,)
         */
        xt::xarray<double> getBodyFdP(const xt::xarray<double>& P) const override;

        /**
         * @brief Calculate the Hessian of the scaling function F(P) = log[sum(exp[k(A P + b)])/len(A)] + 1 
         * w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> d^2F/dPdP, shape: (dim_p, dim_p)
         */
        xt::xarray<double> getBodyFdPdP(const xt::xarray<double>& P) const override;

        /**
         * @brief Calculate the third order derivative of the scaling function 
         * F(P) = log[sum(exp[k(A P + b)])/len(A)] + 1 w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> d^3F/dPdPdP, shape: (dim_p, dim_p, dim_p)
         */
        xt::xarray<double> getBodyFdPdPdP(const xt::xarray<double>& P) const override;
}; 


#endif // LOG_SUM_EXP_3D_HPP
