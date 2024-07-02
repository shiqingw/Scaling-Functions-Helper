#include "scalingFunction3d.hpp"
xt::xarray<double> ScalingFunction3d::getRotationMatrix(const xt::xarray<double>& q) const{

    double qx = q(0), qy = q(1), qz = q(2), qw = q(3);
    xt::xarray<double> R = xt::zeros<double>({3, 3});

    R(0,0) = 2*std::pow(qw, 2) + 2*std::pow(qx, 2) - 1;
    R(0,1) = -2*qw*qz + 2*qx*qy;
    R(0,2) = 2*qw*qy + 2*qx*qz;
    R(1,0) = 2*qw*qz + 2*qx*qy;
    R(1,1) = 2*std::pow(qw, 2) + 2*std::pow(qy, 2) - 1;
    R(1,2) = -2*qw*qx + 2*qy*qz;
    R(2,0) = -2*qw*qy + 2*qx*qz;
    R(2,1) = 2*qw*qx + 2*qy*qz;
    R(2,2) = 2*std::pow(qw, 2) + 2*std::pow(qz, 2) - 1;

    return R;
}

xt::xarray<double> ScalingFunction3d::getBodyP(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                                const xt::xarray<double>& q) const{
    xt::xarray<double> R = getRotationMatrix(q);
    return xt::linalg::dot(xt::transpose(R, {1, 0}), (p - d));
}

xt::xarray<double> ScalingFunction3d::getBodyPdp(const xt::xarray<double>& q) const{
    xt::xarray<double> R = getRotationMatrix(q);
    return xt::transpose(R, {1, 0});
}

xt::xarray<double> ScalingFunction3d::getBodyPdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                                const xt::xarray<double>& q) const{
    
    int dim_p = 3, dim_q = 4;
    int dim_x = dim_p + dim_q;
    double qx = q(0), qy = q(1), qz = q(2), qw = q(3);
    double dx = d(0), dy = d(1), dz = d(2);
    double px = p(0), py = p(1), pz = p(2);
    xt::xarray<double> P_dx = xt::zeros<double>({dim_p, dim_x});

    xt::xarray<double> R = getRotationMatrix(q);
    xt::view(P_dx, xt::all(), xt::range(0, dim_p)) = - xt::transpose(R, {1, 0});
    P_dx(0,3) = -4*qx*(dx - px) - 2*qy*(dy - py) - 2*qz*(dz - pz);
    P_dx(0,4) = 2*qw*(dz - pz) - 2*qx*(dy - py);
    P_dx(0,5) = -2*qw*(dy - py) - 2*qx*(dz - pz);
    P_dx(0,6) = -4*qw*(dx - px) + 2*qy*(dz - pz) - 2*qz*(dy - py);
    P_dx(1,3) = -2*qw*(dz - pz) - 2*qy*(dx - px);
    P_dx(1,4) = -2*qx*(dx - px) - 4*qy*(dy - py) - 2*qz*(dz - pz);
    P_dx(1,5) = 2*qw*(dx - px) - 2*qy*(dz - pz);
    P_dx(1,6) = -4*qw*(dy - py) - 2*qx*(dz - pz) + 2*qz*(dx - px);
    P_dx(2,3) = 2*qw*(dy - py) - 2*qz*(dx - px);
    P_dx(2,4) = -2*qw*(dx - px) - 2*qz*(dy - py);
    P_dx(2,5) = -2*qx*(dx - px) - 2*qy*(dy - py) - 4*qz*(dz - pz);
    P_dx(2,6) = -4*qw*(dz - pz) + 2*qx*(dy - py) - 2*qy*(dx - px);

    return P_dx;
}

xt::xarray<double> ScalingFunction3d::getBodyPdpdx(const xt::xarray<double>& q) const{

    int dim_p = 3, dim_q = 4;
    int dim_x = dim_p + dim_q;
    double qx = q(0), qy = q(1), qz = q(2), qw = q(3);
    xt::xarray<double> P_dpdx = xt::zeros<double>({dim_p, dim_p, dim_x});

    P_dpdx(0,0,3) = 4*qx;
    P_dpdx(0,0,6) = 4*qw;
    P_dpdx(0,1,3) = 2*qy;
    P_dpdx(0,1,4) = 2*qx;
    P_dpdx(0,1,5) = 2*qw;
    P_dpdx(0,1,6) = 2*qz;
    P_dpdx(0,2,3) = 2*qz;
    P_dpdx(0,2,4) = -2*qw;
    P_dpdx(0,2,5) = 2*qx;
    P_dpdx(0,2,6) = -2*qy;
    P_dpdx(1,0,3) = 2*qy;
    P_dpdx(1,0,4) = 2*qx;
    P_dpdx(1,0,5) = -2*qw;
    P_dpdx(1,0,6) = -2*qz;
    P_dpdx(1,1,4) = 4*qy;
    P_dpdx(1,1,6) = 4*qw;
    P_dpdx(1,2,3) = 2*qw;
    P_dpdx(1,2,4) = 2*qz;
    P_dpdx(1,2,5) = 2*qy;
    P_dpdx(1,2,6) = 2*qx;
    P_dpdx(2,0,3) = 2*qz;
    P_dpdx(2,0,4) = 2*qw;
    P_dpdx(2,0,5) = 2*qx;
    P_dpdx(2,0,6) = 2*qy;
    P_dpdx(2,1,3) = -2*qw;
    P_dpdx(2,1,4) = 2*qz;
    P_dpdx(2,1,5) = 2*qy;
    P_dpdx(2,1,6) = -2*qx;
    P_dpdx(2,2,5) = 4*qz;
    P_dpdx(2,2,6) = 4*qw;

    return P_dpdx;
}

xt::xarray<double> ScalingFunction3d::getBodyPdxdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                                    const xt::xarray<double>& q) const{

    int dim_p = 3, dim_q = 4;
    int dim_x = dim_p + dim_q;
    double qx = q(0), qy = q(1), qz = q(2), qw = q(3);
    double dx = d(0), dy = d(1), dz = d(2);
    double px = p(0), py = p(1), pz = p(2);
    xt::xarray<double> P_dxdx = xt::zeros<double>({dim_p, dim_x, dim_x});

    P_dxdx(0,0,3) = -4*qx;
    P_dxdx(0,0,6) = -4*qw;
    P_dxdx(0,1,3) = -2*qy;
    P_dxdx(0,1,4) = -2*qx;
    P_dxdx(0,1,5) = -2*qw;
    P_dxdx(0,1,6) = -2*qz;
    P_dxdx(0,2,3) = -2*qz;
    P_dxdx(0,2,4) = 2*qw;
    P_dxdx(0,2,5) = -2*qx;
    P_dxdx(0,2,6) = 2*qy;
    P_dxdx(0,3,0) = -4*qx;
    P_dxdx(0,3,1) = -2*qy;
    P_dxdx(0,3,2) = -2*qz;
    P_dxdx(0,3,3) = -4*dx + 4*px;
    P_dxdx(0,3,4) = -2*dy + 2*py;
    P_dxdx(0,3,5) = -2*dz + 2*pz;
    P_dxdx(0,4,1) = -2*qx;
    P_dxdx(0,4,2) = 2*qw;
    P_dxdx(0,4,3) = -2*dy + 2*py;
    P_dxdx(0,4,6) = 2*dz - 2*pz;
    P_dxdx(0,5,1) = -2*qw;
    P_dxdx(0,5,2) = -2*qx;
    P_dxdx(0,5,3) = -2*dz + 2*pz;
    P_dxdx(0,5,6) = -2*dy + 2*py;
    P_dxdx(0,6,0) = -4*qw;
    P_dxdx(0,6,1) = -2*qz;
    P_dxdx(0,6,2) = 2*qy;
    P_dxdx(0,6,4) = 2*dz - 2*pz;
    P_dxdx(0,6,5) = -2*dy + 2*py;
    P_dxdx(0,6,6) = -4*dx + 4*px;
    P_dxdx(1,0,3) = -2*qy;
    P_dxdx(1,0,4) = -2*qx;
    P_dxdx(1,0,5) = 2*qw;
    P_dxdx(1,0,6) = 2*qz;
    P_dxdx(1,1,4) = -4*qy;
    P_dxdx(1,1,6) = -4*qw;
    P_dxdx(1,2,3) = -2*qw;
    P_dxdx(1,2,4) = -2*qz;
    P_dxdx(1,2,5) = -2*qy;
    P_dxdx(1,2,6) = -2*qx;
    P_dxdx(1,3,0) = -2*qy;
    P_dxdx(1,3,2) = -2*qw;
    P_dxdx(1,3,4) = -2*dx + 2*px;
    P_dxdx(1,3,6) = -2*dz + 2*pz;
    P_dxdx(1,4,0) = -2*qx;
    P_dxdx(1,4,1) = -4*qy;
    P_dxdx(1,4,2) = -2*qz;
    P_dxdx(1,4,3) = -2*dx + 2*px;
    P_dxdx(1,4,4) = -4*dy + 4*py;
    P_dxdx(1,4,5) = -2*dz + 2*pz;
    P_dxdx(1,5,0) = 2*qw;
    P_dxdx(1,5,2) = -2*qy;
    P_dxdx(1,5,4) = -2*dz + 2*pz;
    P_dxdx(1,5,6) = 2*dx - 2*px;
    P_dxdx(1,6,0) = 2*qz;
    P_dxdx(1,6,1) = -4*qw;
    P_dxdx(1,6,2) = -2*qx;
    P_dxdx(1,6,3) = -2*dz + 2*pz;
    P_dxdx(1,6,5) = 2*dx - 2*px;
    P_dxdx(1,6,6) = -4*dy + 4*py;
    P_dxdx(2,0,3) = -2*qz;
    P_dxdx(2,0,4) = -2*qw;
    P_dxdx(2,0,5) = -2*qx;
    P_dxdx(2,0,6) = -2*qy;
    P_dxdx(2,1,3) = 2*qw;
    P_dxdx(2,1,4) = -2*qz;
    P_dxdx(2,1,5) = -2*qy;
    P_dxdx(2,1,6) = 2*qx;
    P_dxdx(2,2,5) = -4*qz;
    P_dxdx(2,2,6) = -4*qw;
    P_dxdx(2,3,0) = -2*qz;
    P_dxdx(2,3,1) = 2*qw;
    P_dxdx(2,3,5) = -2*dx + 2*px;
    P_dxdx(2,3,6) = 2*dy - 2*py;
    P_dxdx(2,4,0) = -2*qw;
    P_dxdx(2,4,1) = -2*qz;
    P_dxdx(2,4,5) = -2*dy + 2*py;
    P_dxdx(2,4,6) = -2*dx + 2*px;
    P_dxdx(2,5,0) = -2*qx;
    P_dxdx(2,5,1) = -2*qy;
    P_dxdx(2,5,2) = -4*qz;
    P_dxdx(2,5,3) = -2*dx + 2*px;
    P_dxdx(2,5,4) = -2*dy + 2*py;
    P_dxdx(2,5,5) = -4*dz + 4*pz;
    P_dxdx(2,6,0) = -2*qy;
    P_dxdx(2,6,1) = 2*qx;
    P_dxdx(2,6,2) = -4*qw;
    P_dxdx(2,6,3) = 2*dy - 2*py;
    P_dxdx(2,6,4) = -2*dx + 2*px;
    P_dxdx(2,6,6) = -4*dz + 4*pz;

    return P_dxdx;
}

xt::xarray<double> ScalingFunction3d::getBodyPdpdxdx() const{
    
    int dim_p = 3, dim_q = 4;
    int dim_x = dim_p + dim_q;
    xt::xarray<double> P_dpdxdx = xt::zeros<double>({dim_p, dim_p, dim_x, dim_x});

    P_dpdxdx(0,0,3,3) = 4;
    P_dpdxdx(0,0,6,6) = 4;
    P_dpdxdx(0,1,3,4) = 2;
    P_dpdxdx(0,1,4,3) = 2;
    P_dpdxdx(0,1,5,6) = 2;
    P_dpdxdx(0,1,6,5) = 2;
    P_dpdxdx(0,2,3,5) = 2;
    P_dpdxdx(0,2,4,6) = -2;
    P_dpdxdx(0,2,5,3) = 2;
    P_dpdxdx(0,2,6,4) = -2;
    P_dpdxdx(1,0,3,4) = 2;
    P_dpdxdx(1,0,4,3) = 2;
    P_dpdxdx(1,0,5,6) = -2;
    P_dpdxdx(1,0,6,5) = -2;
    P_dpdxdx(1,1,4,4) = 4;
    P_dpdxdx(1,1,6,6) = 4;
    P_dpdxdx(1,2,3,6) = 2;
    P_dpdxdx(1,2,4,5) = 2;
    P_dpdxdx(1,2,5,4) = 2;
    P_dpdxdx(1,2,6,3) = 2;
    P_dpdxdx(2,0,3,5) = 2;
    P_dpdxdx(2,0,4,6) = 2;
    P_dpdxdx(2,0,5,3) = 2;
    P_dpdxdx(2,0,6,4) = 2;
    P_dpdxdx(2,1,3,6) = -2;
    P_dpdxdx(2,1,4,5) = 2;
    P_dpdxdx(2,1,5,4) = 2;
    P_dpdxdx(2,1,6,3) = -2;
    P_dpdxdx(2,2,5,5) = 4;
    P_dpdxdx(2,2,6,6) = 4;

    return P_dpdxdx;
}

double ScalingFunction3d::getWorldF(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        const xt::xarray<double>& q) const{
    
    xt::xarray<double> P = getBodyP(p, d, q); // shape (dim_p,)
    return getBodyF(P);
}

xt::xarray<double> ScalingFunction3d::getWorldFdp(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        const xt::xarray<double>& q) const{
    
    xt::xarray<double> P = getBodyP(p, d, q); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(q); // shape (dim_p, dim_p)

    xt::xarray<double> F_dP = getBodyFdP(P); // shape (dim_p, )
    xt::xarray<double> F_dp = xt::linalg::dot(F_dP, P_dp); // shape (dim_p, )

    return F_dp;
}

xt::xarray<double> ScalingFunction3d::getWorldFdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        const xt::xarray<double>& q) const{
    
    if (isMoving == false){
        return xt::zeros<double>({7});
    }

    xt::xarray<double> P = getBodyP(p, d, q); // shape (dim_p, )
    xt::xarray<double> P_dx = getBodyPdx(p, d, q); // shape (dim_p, dim_x)

    xt::xarray<double> F_dP = getBodyFdP(P); // shape (dim_p, )
    xt::xarray<double> F_dx = xt::linalg::dot(F_dP, P_dx); // shape (dim_x, )

    return F_dx;
}

xt::xarray<double> ScalingFunction3d::getWorldFdpdp(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        const xt::xarray<double>& q) const{
    
    xt::xarray<double> P = getBodyP(p, d, q); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(q); // shape (dim_p, dim_p)

    xt::xarray<double> F_dPdP = getBodyFdPdP(P); // shape (dim_p, dim_p)
    xt::xarray<double> F_dpdp = xt::linalg::dot(xt::transpose(P_dp, {1,0}), xt::linalg::dot(F_dPdP, P_dp)); // shape (dim_p, dim_p)

    return F_dpdp;
}

xt::xarray<double> ScalingFunction3d::getWorldFdpdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        const xt::xarray<double>& q) const{
    
    if (isMoving == false){
        return xt::zeros<double>({3,7});
    }

    xt::xarray<double> P = getBodyP(p, d, q); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(q); // shape (dim_p, dim_p)
    xt::xarray<double> P_dx = getBodyPdx(p, d, q); // shape (dim_p, dim_x)
    xt::xarray<double> P_dpdx = getBodyPdpdx(q); // shape (dim_p, dim_p, dim_x)

    xt::xarray<double> F_dP = getBodyFdP(P); // shape (dim_p, )
    xt::xarray<double> F_dPdP = getBodyFdPdP(P); // shape (dim_p, dim_p)

    xt::xarray<double> F_dpdx = xt::linalg::tensordot(F_dP, P_dpdx, {0}, {0}); // shape (dim_p, dim_x)
    F_dpdx += xt::linalg::dot(xt::transpose(P_dp, {1,0}), xt::linalg::dot(F_dPdP, P_dx)); // shape (dim_p, dim_x)

    return F_dpdx;
}

xt::xarray<double> ScalingFunction3d::getWorldFdxdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        const xt::xarray<double>& q) const{
    
    if (isMoving == false){
        return xt::zeros<double>({7,7});
    }

    xt::xarray<double> P = getBodyP(p, d, q); // shape (dim_p, )
    xt::xarray<double> P_dx = getBodyPdx(p, d, q); // shape (dim_p, dim_x)
    xt::xarray<double> P_dxdx = getBodyPdxdx(p, d, q); // shape (dim_p, dim_x, dim_x)

    xt::xarray<double> F_dP = getBodyFdP(P); // shape (dim_p, )
    xt::xarray<double> F_dPdP = getBodyFdPdP(P); // shape (dim_p, dim_p)

    xt::xarray<double> F_dxdx = xt::linalg::tensordot(F_dP, P_dxdx, {0}, {0}); // shape (dim_x, dim_x)
    F_dxdx += xt::linalg::dot(xt::transpose(P_dx, {1,0}), xt::linalg::dot(F_dPdP, P_dx)); // shape (dim_x, dim_x)

    return F_dxdx;
}

xt::xarray<double> ScalingFunction3d::getWorldFdpdpdp(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        const xt::xarray<double>& q) const{
                            
    xt::xarray<double> P = getBodyP(p, d, q); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(q); // shape (dim_p, dim_p)

    xt::xarray<double> F_dPdPdP = getBodyFdPdPdP(P); // shape (dim_p, dim_p, dim_p)

    xt::xarray<double> F_dpdpdp = xt::linalg::tensordot(F_dPdPdP, P_dp, {2}, {0}); // shape (dim_p, dim_p, dim_p)
    F_dpdpdp = xt::linalg::tensordot(xt::transpose(P_dp, {1,0}), F_dpdpdp, {1}, {0}); // shape (dim_p, dim_p, dim_p)
    F_dpdpdp = xt::transpose(F_dpdpdp, {0,2,1});
    F_dpdpdp = xt::linalg::tensordot(F_dpdpdp, P_dp, {2}, {0}); // shape (dim_p, dim_p, dim_p)
    F_dpdpdp = xt::transpose(F_dpdpdp, {0,2,1});

    return F_dpdpdp;
}

xt::xarray<double> ScalingFunction3d::getWorldFdpdpdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        const xt::xarray<double>& q) const{
    
    if (isMoving == false){
        return xt::zeros<double>({3,3,7});
    }

    xt::xarray<double> P = getBodyP(p, d, q); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(q); // shape (dim_p, dim_p)
    xt::xarray<double> P_dx = getBodyPdx(p, d, q); // shape (dim_p, dim_x)
    xt::xarray<double> P_dpdx = getBodyPdpdx(q); // shape (dim_p, dim_p, dim_x)

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

xt::xarray<double> ScalingFunction3d::getWorldFdpdxdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                        const xt::xarray<double>& q) const{
    
    if (isMoving == false){
        return xt::zeros<double>({3,7,7});
    }
    xt::xarray<double> P = getBodyP(p, d, q); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(q); // shape (dim_p, dim_p)
    xt::xarray<double> P_dx = getBodyPdx(p, d, q); // shape (dim_p, dim_x)
    xt::xarray<double> P_dpdx = getBodyPdpdx(q); // shape (dim_p, dim_p, dim_x)
    xt::xarray<double> P_dxdx = getBodyPdxdx(p, d, q); // shape (dim_p, dim_x, dim_x)
    xt::xarray<double> P_dpdxdx = getBodyPdpdxdx(); // shape (dim_p, dim_p, dim_x, dim_x)

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

std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>>
    ScalingFunction3d::getWorldFFirstToSecondDers(const xt::xarray<double>& p, const xt::xarray<double>& d, const xt::xarray<double>& q) const {

    int dim_p = 3, dim_x = 7;

    xt::xarray<double> P = getBodyP(p, d, q); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(q); // shape (dim_p, dim_p)
    xt::xarray<double> P_dx = getBodyPdx(p, d, q); // shape (dim_p, dim_x)
    xt::xarray<double> P_dpdx = getBodyPdpdx(q); // shape (dim_p, dim_p, dim_x)

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

std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>>
    ScalingFunction3d::getWorldFFirstToThirdDers(const xt::xarray<double>& p, const xt::xarray<double>& d, const xt::xarray<double>& q) const {

    int dim_p = 3, dim_x = 7;

    xt::xarray<double> P = getBodyP(p, d, q); // shape (dim_p, )
    xt::xarray<double> P_dp = getBodyPdp(q); // shape (dim_p, dim_p)
    xt::xarray<double> P_dx = getBodyPdx(p, d, q); // shape (dim_p, dim_x)
    xt::xarray<double> P_dpdx = getBodyPdpdx(q); // shape (dim_p, dim_p, dim_x)
    xt::xarray<double> P_dxdx = getBodyPdxdx(p, d, q); // shape (dim_p, dim_x, dim_x)
    xt::xarray<double> P_dpdxdx = getBodyPdpdxdx(); // shape (dim_p, dim_p, dim_x, dim_x)

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