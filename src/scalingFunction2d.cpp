#include "scalingFunction2d.hpp"
xt::xarray<double> ScalingFunction2d::getRotationMatrix(double theta) const{
    xt::xarray<double> R = {{std::cos(theta), -std::sin(theta)},
                            {std::sin(theta), std::cos(theta)}};
    return R;
}

xt::xarray<double> ScalingFunction2d::getBodyP(const xt::xarray<double>& p, const xt::xarray<double>& d, double theta) const{
    xt::xarray<double> R = getRotationMatrix(theta);
    return xt::linalg::dot(xt::transpose(R, {1, 0}), (p - d));
}

xt::xarray<double> ScalingFunction2d::getBodyPdp(double theta) const{
    xt::xarray<double> R = getRotationMatrix(theta);
    return xt::transpose(R, {1, 0});
}

xt::xarray<double> ScalingFunction2d::getBodyPdx(const xt::xarray<double>& p, const xt::xarray<double>& d, double theta) const{
    
    int dim_p = 2, dim_x = 3;
    double dx = d(0), dy = d(1);
    double px = p(0), py = p(1);
    xt::xarray<double> P_dx = xt::zeros<double>({dim_p, dim_x});

    P_dx(0,0) = -std::cos(theta);
    P_dx(0,1) = -std::sin(theta);
    P_dx(0,2) = -(-dx + px)*std::sin(theta) + (-dy + py)*std::cos(theta);
    P_dx(1,0) = std::sin(theta);
    P_dx(1,1) = -std::cos(theta);
    P_dx(1,2) = -(-dx + px)*std::cos(theta) - (-dy + py)*std::sin(theta);

    return P_dx;
}

xt::xarray<double> ScalingFunction2d::getBodyPdpdx(double theta) const{

    int dim_p = 2, dim_x = 3;
    xt::xarray<double> P_dpdx = xt::zeros<double>({dim_p, dim_p, dim_x});

    P_dpdx(0,0,2) = -std::sin(theta);
    P_dpdx(0,1,2) = std::cos(theta);
    P_dpdx(1,0,2) = -std::cos(theta);
    P_dpdx(1,1,2) = -std::sin(theta);

    return P_dpdx;
}

xt::xarray<double> ScalingFunction2d::getBodyPdxdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                                    double theta) const{

    int dim_p = 2, dim_x = 3;
    double dx = d(0), dy = d(1);
    double px = p(0), py = p(1);
    xt::xarray<double> P_dxdx = xt::zeros<double>({dim_p, dim_x, dim_x});

    P_dxdx(0,0,2) = std::sin(theta);
    P_dxdx(0,1,2) = -std::cos(theta);
    P_dxdx(0,2,0) = std::sin(theta);
    P_dxdx(0,2,1) = -std::cos(theta);
    P_dxdx(0,2,2) = -(-dx + px)*std::cos(theta) - (-dy + py)*std::sin(theta);
    P_dxdx(1,0,2) = std::cos(theta);
    P_dxdx(1,1,2) = std::sin(theta);
    P_dxdx(1,2,0) = std::cos(theta);
    P_dxdx(1,2,1) = std::sin(theta);
    P_dxdx(1,2,2) = (-dx + px)*std::sin(theta) - (-dy + py)*std::cos(theta);

    return P_dxdx;
}

xt::xarray<double> ScalingFunction2d::getBodyPdpdxdx(double theta) const{
    
    int dim_p = 2, dim_x = 3;
    xt::xarray<double> P_dpdxdx = xt::zeros<double>({dim_p, dim_p, dim_x, dim_x});

    P_dpdxdx(0,0,2,2) = -std::cos(theta);
    P_dpdxdx(0,1,2,2) = -std::sin(theta);
    P_dpdxdx(1,0,2,2) = std::sin(theta);
    P_dpdxdx(1,1,2,2) = -std::cos(theta);

    return P_dpdxdx;
}

double ScalingFunction2d::getWorldF(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const{
    
    xt::xarray<double> P = getBodyP(p, d, theta); // shape (dim_p, )
    return getBodyF(P);
}

xt::xarray<double> ScalingFunction2d::getWorldFdp(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const{
    
    xt::xarray<double> P = getBodyP(p, d, theta); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(theta); // shape (dim_p, dim_p)

    xt::xarray<double> F_dP = getBodyFdP(P); // shape (dim_p, )
    xt::xarray<double> F_dp = xt::linalg::dot(F_dP, P_dp); // shape (dim_p, )

    return F_dp;
}

xt::xarray<double> ScalingFunction2d::getWorldFdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const{

    if (isMoving == false){
        return xt::zeros<double>({3});
    }

    xt::xarray<double> P = getBodyP(p, d, theta); // shape (dim_p, )
    xt::xarray<double> P_dx = getBodyPdx(p, d, theta); // shape (dim_p, dim_x)

    xt::xarray<double> F_dP = getBodyFdP(P); // shape (dim_p, )
    xt::xarray<double> F_dx = xt::linalg::dot(F_dP, P_dx); // shape (dim_x, )

    return F_dx;
}

xt::xarray<double> ScalingFunction2d::getWorldFdpdp(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const{
    
    xt::xarray<double> P = getBodyP(p, d, theta); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(theta); // shape (dim_p, dim_p)

    xt::xarray<double> F_dPdP = getBodyFdPdP(P); // shape (dim_p, dim_p)
    xt::xarray<double> F_dpdp = xt::linalg::dot(xt::transpose(P_dp, {1,0}), xt::linalg::dot(F_dPdP, P_dp)); // shape (dim_p, dim_p)

    return F_dpdp;
}

xt::xarray<double> ScalingFunction2d::getWorldFdpdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const{
    
    if (isMoving == false){
        return xt::zeros<double>({2,3});
    }

    xt::xarray<double> P = getBodyP(p, d, theta); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(theta); // shape (dim_p, dim_p)
    xt::xarray<double> P_dx = getBodyPdx(p, d, theta); // shape (dim_p, dim_x)
    xt::xarray<double> P_dpdx = getBodyPdpdx(theta); // shape (dim_p, dim_p, dim_x)

    xt::xarray<double> F_dP = getBodyFdP(P); // shape (dim_p, )
    xt::xarray<double> F_dPdP = getBodyFdPdP(P); // shape (dim_p, dim_p)

    xt::xarray<double> F_dpdx = xt::linalg::tensordot(F_dP, P_dpdx, {0}, {0}); // shape (dim_p, dim_x)
    F_dpdx += xt::linalg::dot(xt::transpose(P_dp, {1,0}), xt::linalg::dot(F_dPdP, P_dx)); // shape (dim_p, dim_x)

    return F_dpdx;
}

xt::xarray<double> ScalingFunction2d::getWorldFdxdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const{
    
    if (isMoving == false){
        return xt::zeros<double>({3,3});
    }

    xt::xarray<double> P = getBodyP(p, d, theta); // shape (dim_p, )
    xt::xarray<double> P_dx = getBodyPdx(p, d, theta); // shape (dim_p, dim_x)
    xt::xarray<double> P_dxdx = getBodyPdxdx(p, d, theta); // shape (dim_p, dim_x, dim_x)

    xt::xarray<double> F_dP = getBodyFdP(P); // shape (dim_p, )
    xt::xarray<double> F_dPdP = getBodyFdPdP(P); // shape (dim_p, dim_p)

    xt::xarray<double> F_dxdx = xt::linalg::tensordot(F_dP, P_dxdx, {0}, {0}); // shape (dim_x, dim_x)
    F_dxdx += xt::linalg::dot(xt::transpose(P_dx, {1,0}), xt::linalg::dot(F_dPdP, P_dx)); // shape (dim_x, dim_x)

    return F_dxdx;
}

xt::xarray<double> ScalingFunction2d::getWorldFdpdpdp(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const{
                            
    xt::xarray<double> P = getBodyP(p, d, theta); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(theta); // shape (dim_p, dim_p)

    xt::xarray<double> F_dPdPdP = getBodyFdPdPdP(P); // shape (dim_p, dim_p, dim_p)

    xt::xarray<double> F_dpdpdp = xt::linalg::tensordot(F_dPdPdP, P_dp, {2}, {0}); // shape (dim_p, dim_p, dim_p)
    F_dpdpdp = xt::linalg::tensordot(xt::transpose(P_dp, {1,0}), F_dpdpdp, {1}, {0}); // shape (dim_p, dim_p, dim_p)
    F_dpdpdp = xt::transpose(F_dpdpdp, {0,2,1});
    F_dpdpdp = xt::linalg::tensordot(F_dpdpdp, P_dp, {2}, {0}); // shape (dim_p, dim_p, dim_p)
    F_dpdpdp = xt::transpose(F_dpdpdp, {0,2,1});

    return F_dpdpdp;
}

xt::xarray<double> ScalingFunction2d::getWorldFdpdpdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const{
    
    if (isMoving == false){
        return xt::zeros<double>({2,2,3});
    }

    xt::xarray<double> P = getBodyP(p, d, theta); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(theta); // shape (dim_p, dim_p)
    xt::xarray<double> P_dx = getBodyPdx(p, d, theta); // shape (dim_p, dim_x)
    xt::xarray<double> P_dpdx = getBodyPdpdx(theta); // shape (dim_p, dim_p, dim_x)

    xt::xarray<double> F_dPdP = getBodyFdPdP(P); // shape (dim_p, dim_p)
    xt::xarray<double> F_dPdPdP = getBodyFdPdPdP(P); // shape (dim_p, dim_p, dim_p)

    xt::xarray<double> F_dpdpdx_1 = xt::linalg::tensordot(xt::linalg::dot(xt::transpose(P_dp, {1,0}), 
        F_dPdP), P_dpdx, {1}, {0}); // shape (dim_p, dim_p, dim_x)
    F_dpdpdx_1 += xt::transpose(F_dpdpdx_1, {1,0,2}); // shape (dim_p, dim_p, dim_x)

    xt::xarray<double> F_dpdpdx_2 = xt::linalg::tensordot(xt::transpose(P_dp, {1,0}), 
        xt::linalg::tensordot(F_dPdPdP, P_dx, {2}, {0}), {1}, {0}); // shape (dim_p, dim_p, dim_x)
    F_dpdpdx_2 = xt::transpose(F_dpdpdx_2, {0,2,1}); // shape (dim_p, dim_x, dim_p)
    F_dpdpdx_2 = xt::linalg::tensordot(F_dpdpdx_2, P_dp, {2}, {0}); // shape (dim_p, dim_x, dim_p)
    F_dpdpdx_2 = xt::transpose(F_dpdpdx_2, {0,2,1}); // shape (dim_p, dim_p, dim_x)

    xt::xarray<double> F_dpdpdx = F_dpdpdx_1 + F_dpdpdx_2; // shape (dim_p, dim_p, dim_x)
    return F_dpdpdx;
}

xt::xarray<double> ScalingFunction2d::getWorldFdpdxdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        double theta) const{
    
    if (isMoving == false){
        return xt::zeros<double>({2,3,3});
    }
    xt::xarray<double> P = getBodyP(p, d, theta); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(theta); // shape (dim_p, dim_p)
    xt::xarray<double> P_dx = getBodyPdx(p, d, theta); // shape (dim_p, dim_x)
    xt::xarray<double> P_dpdx = getBodyPdpdx(theta); // shape (dim_p, dim_p, dim_x)
    xt::xarray<double> P_dxdx = getBodyPdxdx(p, d, theta); // shape (dim_p, dim_x, dim_x)
    xt::xarray<double> P_dpdxdx = getBodyPdpdxdx(theta); // shape (dim_p, dim_p, dim_x, dim_x)

    xt::xarray<double> F_dP = getBodyFdP(P); // shape (dim_p,)
    xt::xarray<double> F_dPdP = getBodyFdPdP(P); // shape (dim_p, dim_p)
    xt::xarray<double> F_dPdPdP = getBodyFdPdPdP(P); // shape (dim_p, dim_p, dim_p)

    xt::xarray<double> F_dpdxdx_1 = xt::linalg::tensordot(F_dP, P_dpdxdx, {0}, {0}); // shape (dim_p, dim_x, dim_x)

    xt::xarray<double> F_dpdxdx_2 = xt::transpose(xt::linalg::dot(F_dPdP, P_dx), {1,0}); // shape (dim_x, dim_p)
    F_dpdxdx_2 = xt::linalg::tensordot(F_dpdxdx_2, P_dpdx, {1}, {0}); // shape (dim_x, dim_p, dim_x)
    F_dpdxdx_2 = xt::transpose(F_dpdxdx_2, {1,0,2}); // shape (dim_p, dim_x, dim_x)
    F_dpdxdx_2 += xt::transpose(F_dpdxdx_2, {0,2,1}); // shape (dim_p, dim_x, dim_x)

    xt::xarray<double> F_dpdxdx_3 = xt::linalg::tensordot(xt::linalg::dot(xt::transpose(P_dp, {1,0}), 
        F_dPdP), P_dxdx, {1}, {0}); // shape (dim_p, dim_x, dim_x)

    xt::xarray<double> F_dpdxdx_4 = xt::linalg::tensordot(xt::transpose(P_dp, {1,0}), 
        xt::linalg::tensordot(F_dPdPdP, P_dx, {2}, {0}), {1}, {0}); // shape (dim_p, dim_p, dim_x)
    F_dpdxdx_4 = xt::transpose(F_dpdxdx_4, {0,2,1}); // shape (dim_p, dim_x, dim_p)
    F_dpdxdx_4 = xt::linalg::tensordot(F_dpdxdx_4, P_dx, {2}, {0}); // shape (dim_p, dim_x, dim_x)
    F_dpdxdx_4 = xt::transpose(F_dpdxdx_4, {0,2,1}); // shape (dim_p, dim_x, dim_x)

    xt::xarray<double> F_dpdxdx = F_dpdxdx_1 + F_dpdxdx_2 + F_dpdxdx_3 + F_dpdxdx_4; // shape (dim_p, dim_x, dim_x)

    return F_dpdxdx;
}

std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> ScalingFunction2d::getWorldFFirstToSecondDers(
    const xt::xarray<double>& p, const xt::xarray<double>& d, double theta) const {
    
    int dim_p = 2, dim_x = 3;
    xt::xarray<double> P = getBodyP(p, d, theta); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(theta); // shape (dim_p, dim_p)
    xt::xarray<double> P_dx = getBodyPdx(p, d, theta); // shape (dim_p, dim_x)
    xt::xarray<double> P_dpdx = getBodyPdpdx(theta); // shape (dim_p, dim_p, dim_x)

    // F_dp
    xt::xarray<double> F_dP = getBodyFdP(P); // shape (dim_p, )
    xt::xarray<double> F_dp = xt::linalg::dot(F_dP, P_dp); // shape (dim_p, )

    // F_dx
    xt::xarray<double> F_dx = xt::zeros<double>({dim_x});
    if (isMoving == true){
        F_dx = xt::linalg::dot(F_dP, P_dx); // shape (dim_x, )
    }

    // F_dpdp
    xt::xarray<double> F_dPdP = getBodyFdPdP(P); // shape (dim_p, dim_p)
    xt::xarray<double> F_dpdp = xt::linalg::dot(xt::transpose(P_dp, {1,0}), xt::linalg::dot(F_dPdP, P_dp)); // shape (dim_p, dim_p)

    // F_dpdx
    xt::xarray<double> F_dpdx = xt::zeros<double>({dim_p, dim_x});
    if (isMoving == true){
        F_dpdx = xt::linalg::tensordot(F_dP, P_dpdx, {0}, {0}); // shape (dim_p, dim_x)
        F_dpdx += xt::linalg::dot(xt::transpose(P_dp, {1,0}), xt::linalg::dot(F_dPdP, P_dx)); // shape (dim_p, dim_x)
    }

    return std::make_tuple(F_dp, F_dx, F_dpdp, F_dpdx);
}

std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, 
    xt::xarray<double>> ScalingFunction2d::getWorldFFirstToThirdDers(const xt::xarray<double>& p, const xt::xarray<double>& d, double theta) const {
    
    int dim_p = 2, dim_x = 3;
    xt::xarray<double> P = getBodyP(p, d, theta); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(theta); // shape (dim_p, dim_p)
    xt::xarray<double> P_dx = getBodyPdx(p, d, theta); // shape (dim_p, dim_x)
    xt::xarray<double> P_dpdx = getBodyPdpdx(theta); // shape (dim_p, dim_p, dim_x)
    xt::xarray<double> P_dxdx = getBodyPdxdx(p, d, theta); // shape (dim_p, dim_x, dim_x)
    xt::xarray<double> P_dpdxdx = getBodyPdpdxdx(theta); // shape (dim_p, dim_p, dim_x, dim_x)

    // F_dp
    xt::xarray<double> F_dP = getBodyFdP(P); // shape (dim_p, )
    xt::xarray<double> F_dp = xt::linalg::dot(F_dP, P_dp); // shape (dim_p, )

    // F_dx
    xt::xarray<double> F_dx = xt::zeros<double>({dim_x});
    if (isMoving == true){
        F_dx = xt::linalg::dot(F_dP, P_dx); // shape (dim_x, )
    }

    // F_dpdp
    xt::xarray<double> F_dPdP = getBodyFdPdP(P); // shape (dim_p, dim_p)
    xt::xarray<double> F_dpdp = xt::linalg::dot(xt::transpose(P_dp, {1,0}), xt::linalg::dot(F_dPdP, P_dp)); // shape (dim_p, dim_p)

    // F_dpdx
    xt::xarray<double> F_dpdx = xt::zeros<double>({dim_p, dim_x});
    if (isMoving == true){
        F_dpdx = xt::linalg::tensordot(F_dP, P_dpdx, {0}, {0}); // shape (dim_p, dim_x)
        F_dpdx += xt::linalg::dot(xt::transpose(P_dp, {1,0}), xt::linalg::dot(F_dPdP, P_dx)); // shape (dim_p, dim_x)
    }

    // F_dxdx
    xt::xarray<double> F_dxdx = xt::zeros<double>({dim_x, dim_x});
    if (isMoving == true){
        F_dxdx = xt::linalg::tensordot(F_dP, P_dxdx, {0}, {0}); // shape (dim_x, dim_x)
        F_dxdx += xt::linalg::dot(xt::transpose(P_dx, {1,0}), xt::linalg::dot(F_dPdP, P_dx)); // shape (dim_x, dim_x)
    }

    // F_dpdpdp
    xt::xarray<double> F_dPdPdP = getBodyFdPdPdP(P); // shape (dim_p, dim_p, dim_p)
    xt::xarray<double> F_dpdpdp = xt::linalg::tensordot(F_dPdPdP, P_dp, {2}, {0}); // shape (dim_p, dim_p, dim_p)
    F_dpdpdp = xt::linalg::tensordot(xt::transpose(P_dp, {1,0}), F_dpdpdp, {1}, {0}); // shape (dim_p, dim_p, dim_p)
    F_dpdpdp = xt::transpose(F_dpdpdp, {0,2,1});
    F_dpdpdp = xt::linalg::tensordot(F_dpdpdp, P_dp, {2}, {0}); // shape (dim_p, dim_p, dim_p)
    F_dpdpdp = xt::transpose(F_dpdpdp, {0,2,1});

    // F_dpdpdx
    xt::xarray<double> F_dpdpdx = xt::zeros<double>({dim_p, dim_p, dim_x});
    if (isMoving == true){
        xt::xarray<double> F_dpdpdx_1 = xt::linalg::tensordot(xt::linalg::dot(xt::transpose(P_dp, {1,0}), 
            F_dPdP), P_dpdx, {1}, {0}); // shape (dim_p, dim_p, dim_x)
        F_dpdpdx_1 += xt::transpose(F_dpdpdx_1, {1,0,2}); // shape (dim_p, dim_p, dim_x)

        xt::xarray<double> F_dpdpdx_2 = xt::linalg::tensordot(xt::transpose(P_dp, {1,0}), 
            xt::linalg::tensordot(F_dPdPdP, P_dx, {2}, {0}), {1}, {0}); // shape (dim_p, dim_p, dim_x)
        F_dpdpdx_2 = xt::transpose(F_dpdpdx_2, {0,2,1}); // shape (dim_p, dim_x, dim_p)
        F_dpdpdx_2 = xt::linalg::tensordot(F_dpdpdx_2, P_dp, {2}, {0}); // shape (dim_p, dim_x, dim_p)
        F_dpdpdx_2 = xt::transpose(F_dpdpdx_2, {0,2,1}); // shape (dim_p, dim_p, dim_x)

        F_dpdpdx = F_dpdpdx_1 + F_dpdpdx_2; // shape (dim_p, dim_p, dim_x)
    }

    // F_dpdxdx
    xt::xarray<double> F_dpdxdx = xt::zeros<double>({dim_p, dim_x, dim_x});
    if (isMoving == true){
        xt::xarray<double> F_dpdxdx_1 = xt::linalg::tensordot(F_dP, P_dpdxdx, {0}, {0}); // shape (dim_p, dim_x, dim_x)

        xt::xarray<double> F_dpdxdx_2 = xt::transpose(xt::linalg::dot(F_dPdP, P_dx), {1,0}); // shape (dim_x, dim_p)
        F_dpdxdx_2 = xt::linalg::tensordot(F_dpdxdx_2, P_dpdx, {1}, {0}); // shape (dim_x, dim_p, dim_x)
        F_dpdxdx_2 = xt::transpose(F_dpdxdx_2, {1,0,2}); // shape (dim_p, dim_x, dim_x)
        F_dpdxdx_2 += xt::transpose(F_dpdxdx_2, {0,2,1}); // shape (dim_p, dim_x, dim_x)

        xt::xarray<double> F_dpdxdx_3 = xt::linalg::tensordot(xt::linalg::dot(xt::transpose(P_dp, {1,0}), 
            F_dPdP), P_dxdx, {1}, {0}); // shape (dim_p, dim_x, dim_x)

        xt::xarray<double> F_dpdxdx_4 = xt::linalg::tensordot(xt::transpose(P_dp, {1,0}), 
            xt::linalg::tensordot(F_dPdPdP, P_dx, {2}, {0}), {1}, {0}); // shape (dim_p, dim_p, dim_x)
        F_dpdxdx_4 = xt::transpose(F_dpdxdx_4, {0,2,1}); // shape (dim_p, dim_x, dim_p)
        F_dpdxdx_4 = xt::linalg::tensordot(F_dpdxdx_4, P_dx, {2}, {0}); // shape (dim_p, dim_x, dim_x)
        F_dpdxdx_4 = xt::transpose(F_dpdxdx_4, {0,2,1}); // shape (dim_p, dim_x, dim_x)

        F_dpdxdx = F_dpdxdx_1 + F_dpdxdx_2 + F_dpdxdx_3 + F_dpdxdx_4; // shape (dim_p, dim_x, dim_x)
    }

    return std::make_tuple(F_dp, F_dx, F_dpdp, F_dpdx, F_dxdx, F_dpdpdp, F_dpdpdx, F_dpdxdx);
}