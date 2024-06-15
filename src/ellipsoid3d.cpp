#include "ellipsoid3d.hpp"

double Ellipsoid3d::getBodyF(const xt::xarray<double>& P) const{
    return xt::linalg::dot(P-mu, xt::linalg::dot(Q, P-mu))(0);
}

xt::xarray<double> Ellipsoid3d::getBodyFdP(const xt::xarray<double>& P) const{
    return 2 * xt::linalg::dot(Q, P - mu);
}

xt::xarray<double> Ellipsoid3d::getBodyFdPdP(const xt::xarray<double>& P) const{
    return 2 * Q;
}

xt::xarray<double> Ellipsoid3d::getBodyFdPdPdP(const xt::xarray<double>& P) const{
    return xt::zeros<double>({3, 3, 3});
}