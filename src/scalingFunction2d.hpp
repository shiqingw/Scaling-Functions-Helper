#ifndef SCALING_FUNCTION_2D_HPP
#define SCALING_FUNCTION_2D_HPP

#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xnoalias.hpp>

class ScalingFunction2d {
    public:
        bool isMoving = false;

        /**
         * @brief Construct a new ScalingFunction2d object
         * 
         * @param isMoving_ True if the scaling function is moving, false otherwise
         */
        ScalingFunction2d(bool isMoving_) : isMoving(isMoving_) {};
        virtual ~ScalingFunction2d() = default;

        /**
         * @brief Calculate the scaling function F(P).
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return double F(P)
         */
        virtual double getBodyF(const xt::xtensor<double, 1>& P) const = 0;

        /**
         * @brief Calculate the gradient of the scaling function F(P) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 1> dF/dP, shape: (dim_p,)
         */
        virtual xt::xtensor<double, 1> getBodyFdP(const xt::xtensor<double, 1>& P) const = 0;

        /**
         * @brief Calculate the Hessian of the scaling function F(P) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 2> d^2F/dPdP, shape: (dim_p, dim_p)
         */
        virtual xt::xtensor<double, 2> getBodyFdPdP(const xt::xtensor<double, 1>& P) const = 0;

        /**
         * @brief Calculate the third order derivative of the scaling function F(P) w.r.t. P.
         * 
         * @param P Position in the body frame, shape: (dim_p,)
         * @return xt::xtensor<double, 3> d^3F/dPdPdP, shape: (dim_p, dim_p, dim_p)
         */
        virtual xt::xtensor<double, 3> getBodyFdPdPdP(const xt::xtensor<double, 1>& P) const = 0;

        /**
         * @brief Get the rotation matrix R(theta)
         * 
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 2> R, shape: (dim_p, dim_p)
         */
        xt::xtensor<double, 2> getRotationMatrix(double theta) const;

        /**
         * @brief Get the position in body P = R(theta).T (p - d)
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 1> P, shape: (dim_p,)
         */
        xt::xtensor<double, 1> getBodyP(const xt::xtensor<double, 1>& p, const xt::xtensor<double, 1>& d,
                                    double theta) const;

        /**
         * @brief P = R(theta).T (p - d) because p = R(theta) * P + d.
         * 
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 2> P_dp, shape: (dim_p, dim_p)
         */
        xt::xtensor<double, 2> getBodyPdp(double theta) const;

        /**
         * @brief P = R(theta).T (p - d). Get the gradient of P w.r.t. x=[d,theta]. dim_x = dim_p + 1.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 2> P_dx, shape: (dim_p, dim_x)
         */
        xt::xtensor<double, 2> getBodyPdx(const xt::xtensor<double, 1>& p, const xt::xtensor<double, 1>& d, double theta) const;
        
        /**
         * @brief P = R(theta).T (p - d). x=[d,theta]. Get d^2P/dpdx. dim_x = dim_p + 1.
         * 
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 3> P_dxdx, shape: (dim_p, dim_p, dim_x)
         */
        xt::xtensor<double, 3> getBodyPdpdx(double theta) const;

        /**
         * @brief P = R(theta).T (p - d). x=[d,theta]. Get d^2P/dxdx. dim_x = dim_p + 1.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 3> P_dxdx, shape: (dim_p, dim_x, dim_x)
         */
        xt::xtensor<double, 3> getBodyPdxdx(const xt::xtensor<double, 1>& p, const xt::xtensor<double, 1>& d, double theta) const;

        /**
         * @brief P = R(theta).T (p - d). x=[d,theta]. Get d^3P/dpdxdx. dim_x = dim_p + 1.
         * 
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 4> P_dpdx, shape: (dim_p, dim_p, dim_x, dim_x)
         */
        xt::xtensor<double, 4> getBodyPdpdxdx(double theta) const;

        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)].
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return double F(p) = F(P) = F[R(theta).T (p - d)]
         */
        double getWorldF(const xt::xtensor<double, 1>& p, const xt::xtensor<double, 1>& d,
                                        double theta) const;

        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get dF/dp.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 1> F_dp, shape: (dim_p,)
         */
        xt::xtensor<double, 1> getWorldFdp(const xt::xtensor<double, 1>& p, const xt::xtensor<double, 1>& d,
                                        double theta) const;

        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get dF/dx.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 1> F_dx, shape: (dim_x,) 
         */
        xt::xtensor<double, 1> getWorldFdx(const xt::xtensor<double, 1>& p, const xt::xtensor<double, 1>& d,
                                        double theta) const;

        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get d^2F/dpdp.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 2> F_dpdp, shape: (dim_p, dim_p) 
         */
        xt::xtensor<double, 2> getWorldFdpdp(const xt::xtensor<double, 1>& p, const xt::xtensor<double, 1>& d,
                                        double theta) const;
        
        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get d^2F/dpdx.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 2> F_dpdx, shape: (dim_p, dim_x) 
         */
        xt::xtensor<double, 2> getWorldFdpdx(const xt::xtensor<double, 1>& p, const xt::xtensor<double, 1>& d,
                                        double theta) const;

        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get d^2F/dxdx.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 2> F_dxdx, shape: (dim_x, dim_x) 
         */
        xt::xtensor<double, 2> getWorldFdxdx(const xt::xtensor<double, 1>& p, const xt::xtensor<double, 1>& d,
                                        double theta) const;
        
        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get d^3F/dpdpdp.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 3> F_dpdpdp, shape: (dim_p, dim_p, dim_p) 
         */
        xt::xtensor<double, 3> getWorldFdpdpdp(const xt::xtensor<double, 1>& p, const xt::xtensor<double, 1>& d,
                                        double theta) const;
        
        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get d^3F/dpdpdx.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 3> F_dpdpdx, shape: (dim_p, dim_p, dim_x) 
         */
        xt::xtensor<double, 3> getWorldFdpdpdx(const xt::xtensor<double, 1>& p, const xt::xtensor<double, 1>& d,
                                        double theta) const;

        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get d^3F/dpdxdx.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return xt::xtensor<double, 3> F_dpdxdx, shape: (dim_p, dim_x, dim_x) 
         */
        xt::xtensor<double, 3> getWorldFdpdxdx(const xt::xtensor<double, 1>& p, const xt::xtensor<double, 1>& d,
                                        double theta) const;
        
        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get dF/dp, dF/dx, d^2F/dpdp, d^2F/dpdx.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 2>> 
         * in the order of dF/dp, dF/dx, d^2F/dpdp, d^2F/dpdx.
         */
        std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 2>>
            getWorldFFirstToSecondDers(const xt::xtensor<double, 1>& p, const xt::xtensor<double, 1>& d, double theta) const;

        /**
         * @brief In the world frame, F(p) = F(P) = F[R(theta).T (p - d)]. x=[d,theta]. Get dF/dp, dF/dx, d^2F/dpdp, d^2F/dpdx, d^2F/dxdx, d^3F/dpdpdp, d^3F/dpdpdx, d^3F/dpdxdx.
         * 
         * @param p Position in the world frame, shape: (dim_p,)
         * @param d Origin of the body frame in the world frame, shape: (dim_p,)
         * @param theta Rotation angle representing R(theta)
         * @return std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 2>, xt::xtensor<double, 2>, xt::xtensor<double, 3>, xt::xtensor<double, 3>, xt::xtensor<double, 3>>
         * in the order of dF/dp, dF/dx, d^2F/dpdp, d^2F/dpdx, d^2F/dxdx, d^3F/dpdpdp, d^3F/dpdpdx, d^3F/dpdxdx.
         */
        std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 1>, xt::xtensor<double, 2>, xt::xtensor<double, 2>, xt::xtensor<double, 2>, xt::xtensor<double, 3>, xt::xtensor<double, 3>, xt::xtensor<double, 3>>
            getWorldFFirstToThirdDers(const xt::xtensor<double, 1>& p, const xt::xtensor<double, 1>& d, double theta) const;
        
};

#endif // SCALING_FUNCTION_2D_HPP
