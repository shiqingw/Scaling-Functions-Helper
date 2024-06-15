#include "hyperplane2d.hpp"

double Hyperplane2d::getBodyF(const xt::xarray<double>& P) const{
    return xt::linalg::vdot(a, P) + b;
}

xt::xarray<double> Hyperplane2d::getBodyFdP(const xt::xarray<double>& P) const{
    return a;
}

xt::xarray<double> Hyperplane2d::getBodyFdPdP(const xt::xarray<double>& P) const{
    return xt::zeros<double>({2, 2});
}

xt::xarray<double> Hyperplane2d::getBodyFdPdPdP(const xt::xarray<double>& P) const{
    return xt::zeros<double>({2, 2, 2});
}