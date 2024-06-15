#ifndef SCALING_FUNCTION_2D_HPP
#define SCALING_FUNCTION_2D_HPP

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <cmath>

class ScalingFunction2d {
    public:
        bool isMoving = false;
        ScalingFunction2d(bool isMoving_) : isMoving(isMoving_) {};
        virtual ~ScalingFunction2d() = default;

        /**
         * @brief Calculate the scaling function F(P).
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return double F(P)
         */
        virtual double getBodyF(const xt::xarray<double>& P) const = 0;

        /**
         * @brief Calculate the gradient of the scaling function F(P) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> dF/dP, shape: (dim_p,)
         */
        virtual xt::xarray<double> getBodyFdP(const xt::xarray<double>& P) const = 0;

        /**
         * @brief Calculate the Hessian of the scaling function F(P) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> d^2F/dPdP, shape: (dim_p, dim_p)
         */
        virtual xt::xarray<double> getBodyFdPdP(const xt::xarray<double>& P) const = 0;

        /**
         * @brief Calculate the third order derivative of the scaling function F(P) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xarray<double> d^3F/dPdPdP, shape: (dim_p, dim_p, dim_p)
         */
        virtual xt::xarray<double> getBodyFdPdPdP(const xt::xarray<double>& P) const = 0;

        /**
         * @brief Get the rotation matrix R(theta)
         * 
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> R, shape: (dim_p, dim_p)
         */
        xt::xarray<double> getRotationMatrix(double theta) const;

        /**
         * @brief Get the position in body P = R(theta).T (p - d)
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> P, shape: (dim_p,)
         */
        xt::xarray<double> getBodyP(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                    double theta) const;

        /**
         * @brief P = R(theta).T (p - d) because p = R(theta) * P + d.
         * 
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> P_dp, shape: (dim_p, dim_p)
         */
        xt::xarray<double> getBodyPdp(double theta) const;

        /**
         * @brief P = R(theta).T (p - d). Get the gradient of P w.r.t. x=[d,theta]. dim_x = dim_p + 1.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> P_dx, shape: (dim_p, dim_x)
         */
        xt::xarray<double> getBodyPdx(const xt::xarray<double>& p, const xt::xarray<double>& d, double theta) const;
        
        /**
         * @brief P = R(theta).T (p - d). x=[d,theta]. Get d^2P/dpdx. dim_x = dim_p + 1.
         * 
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> P_dxdx, shape: (dim_p, dim_p, dim_x)
         */
        xt::xarray<double> getBodyPdpdx(double theta) const;

        /**
         * @brief P = R(theta).T (p - d). x=[d,theta]. Get d^2P/dxdx. dim_x = dim_p + 1.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> P_dxdx, shape: (dim_p, dim_x, dim_x)
         */
        xt::xarray<double> getBodyPdxdx(const xt::xarray<double>& p, const xt::xarray<double>& d, double theta) const;

        /**
         * @brief P = R(theta).T (p - d). x=[d,theta]. Get d^3P/dpdxdx. dim_x = dim_p + 1.
         * 
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> P_dpdx, shape: (dim_p, dim_p, dim_x, dim_x)
         */
        xt::xarray<double> getBodyPdpdxdx(double theta) const;

        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get dF/dp.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> F_dp, shape: (dim_p,)
         */
        xt::xarray<double> getWorldFdp(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const;

        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get dF/dx.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> F_dx, shape: (dim_x,) 
         */
        xt::xarray<double> getWorldFdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const;

        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get d^2F/dpdp.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> F_dpdp, shape: (dim_p, dim_p) 
         */
        xt::xarray<double> getWorldFdpdp(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const;
        
        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get d^2F/dpdx.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> F_dpdx, shape: (dim_p, dim_x) 
         */
        xt::xarray<double> getWorldFdpdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const;

        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get d^2F/dxdx.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> F_dxdx, shape: (dim_x, dim_x) 
         */
        xt::xarray<double> getWorldFdxdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const;
        
        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get d^3F/dpdpdp.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> F_dpdpdp, shape: (dim_p, dim_p, dim_p) 
         */
        xt::xarray<double> getWorldFdpdpdp(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const;
        
        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get d^3F/dpdpdx.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> F_dpdpdx, shape: (dim_p, dim_p, dim_x) 
         */
        xt::xarray<double> getWorldFdpdpdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const;

        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get d^3F/dpdxdx.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xarray<double> F_dpdxdx, shape: (dim_p, dim_x, dim_x) 
         */
        xt::xarray<double> getWorldFdpdxdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const;
        
};

#endif // SCALING_FUNCTION_2D_HPP
