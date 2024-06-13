#include "ScalingFunction3d.hpp"

xt::xarray<double> ScalingFunction3d::getBodyPdp(const xt::xarray<double>& R){
    return xt::transpose(R, {1, 0});
};

xt::xarray<double> ScalingFunction3d::getBodyPdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                                const xt::xarray<double>& q, const xt::xarray<double>& R){

    int dim_p = 3, dim_q = 4;
    int dim_x = dim_p + dim_q;
    double qx = q(0), qy = q(1), qz = q(2), qw = q(3);
    double dx = d(0), dy = d(1), dz = d(2);
    double px = p(0), py = p(1), pz = p(2);
    xt::xarray<double> P_dx = xt::zeros<double>({dim_p, dim_x});

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
};

xt::xarray<double> ScalingFunction3d::getBodyPdpdx(const xt::xarray<double>& q){

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
};

xt::xarray<double> ScalingFunction3d::getBodyPdxdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
                                                    const xt::xarray<double>& q){

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
};

xt::xarray<double> ScalingFunction3d::getBodyPdpdxdx(){
    
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
};