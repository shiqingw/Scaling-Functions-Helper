#ifndef SCALING_FUNCTION_3D_HPP
#define SCALING_FUNCTION_3D_HPP

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <tuple>

class ScalingFunction3d {
    public:
        ScalingFunction3d() = default;
        ~ScalingFunction3d() = default;

        // /**
        //  * @brief Calculate the scaling function F(P).
        //  * 
        //  * @param P Position in the body frame, shape: (dim_p,)
        //  * @return double F(P)
        //  */
        // virtual double getBodyF(const xt::xarray<double>& P) const = 0;

        // /**
        //  * @brief Calculate the gradient of the scaling function F(P) w.r.t. P.
        //  * 
        //  * @param P Position in the body frame, shape: (dim_p,)
        //  * @return xt::xarray<double> dF/dP, shape: (dim_p,)
        //  */
        // virtual xt::xarray<double> getBodyFdP(const xt::xarray<double>& P) const = 0;

        // /**
        //  * @brief Calculate the Hessian of the scaling function F(P) w.r.t. P.
        //  * 
        //  * @param P Position in the body frame, shape: (dim_p,)
        //  * @return xt::xarray<double> d^2F/dPdP, shape: (dim_p, dim_p)
        //  */
        // virtual xt::xarray<double> getBodyFdPdP(const xt::xarray<double>& P) const = 0;

        // /**
        //  * @brief Calculate the third order derivative of the scaling function F(P) w.r.t. P.
        //  * 
        //  * @param P Position in the body frame, shape: (dim_p,)
        //  * @return xt::xarray<double> d^3F/dPdPdP, shape: (dim_p, dim_p, dim_p)
        //  */
        // virtual xt::xarray<double> getBodyFdPdPdP(const xt::xarray<double>& P) const = 0;

        /**
         * @brief P = R(q).T (p - d) because p = R(q) * P + d.
         * 
         * @param R Rotation matrix (body to world frame), shape: (dim_p, dim_p)
         * @return xt::xarray<double> P_dp, shape: (dim_p, dim_p)
         */
        xt::xarray<double> getBodyPdp(const xt::xarray<double>& R);

        /**
         * @brief P = R(q).T (p - d). Get the gradient of P w.r.t. x=[d,q]. dim_x = dim_p + dim_q.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param q Unit quaternion [qx,qy,qz,qw] representing R(q), shape: (dim_q,)
         * @param R Rotation matrix (body to world frame), shape: (dim_p, dim_p)
         * @return xt::xarray<double> P_dx, shape: (dim_p, dim_x)
         */
        xt::xarray<double> getBodyPdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
            const xt::xarray<double>& q, const xt::xarray<double>& R);
        
        /**
         * @brief P = R(q).T (p - d). x=[d,q]. Get d^2P/dpdx. dim_x = dim_p + dim_q.
         * 
         * @param q Unit quaternion [qx,qy,qz,qw] representing R(q), shape: (dim_q,)
         * @return xt::xarray<double> P_dxdx, shape: (dim_p, dim_p, dim_x)
         */
        xt::xarray<double> getBodyPdpdx(const xt::xarray<double>& q);

        /**
         * @brief P = R(q).T (p - d). x=[d,q]. Get d^2P/dxdx. dim_x = dim_p + dim_q.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param q Unit quaternion [qx,qy,qz,qw] representing R(q), shape: (dim_q,)
         * @return xt::xarray<double> P_dxdx, shape: (dim_p, dim_x, dim_x)
         */
        xt::xarray<double> getBodyPdxdx(const xt::xarray<double>& p, const xt::xarray<double>& d, const xt::xarray<double>& q);

        /**
         * @brief P = R(q).T (p - d). x=[d,q]. Get d^3P/dpdxdx. dim_x = dim_p + dim_q.
         * 
         * @return xt::xarray<double> P_dpdx, shape: (dim_p, dim_p, dim_x, dim_x)
         */
        xt::xarray<double> getBodyPdpdxdx();
        
};

#endif // SCALING_FUNCTION_3D_HPP
