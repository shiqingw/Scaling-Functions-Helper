#include "ellipsoid2d.hpp"

double Ellipsoid2d::getBodyF(const xt::xarray<double>& P) const{
    return xt::linalg::dot(P-mu, xt::linalg::dot(Q, P-mu))(0);
}

xt::xarray<double> Ellipsoid2d::getBodyFdP(const xt::xarray<double>& P) const{
    return 2 * xt::linalg::dot(Q, P - mu);
}

xt::xarray<double> Ellipsoid2d::getBodyFdPdP(const xt::xarray<double>& P) const{
    return 2 * Q;
}

xt::xarray<double> Ellipsoid2d::getBodyFdPdPdP(const xt::xarray<double>& P) const{
    return xt::zeros<double>({2, 2, 2});
}