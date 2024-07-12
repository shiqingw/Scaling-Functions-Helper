#include "hyperplane3d.hpp"

double Hyperplane3d::getBodyF(const xt::xarray<double>& P) const{
    return xt::linalg::vdot(a, P) + b + 1;
}

xt::xarray<double> Hyperplane3d::getBodyFdP(const xt::xarray<double>& P) const{
    return a;
}

xt::xarray<double> Hyperplane3d::getBodyFdPdP(const xt::xarray<double>& P) const{
    return xt::zeros<double>({3, 3});
}

xt::xarray<double> Hyperplane3d::getBodyFdPdPdP(const xt::xarray<double>& P) const{
    return xt::zeros<double>({3, 3, 3});
}