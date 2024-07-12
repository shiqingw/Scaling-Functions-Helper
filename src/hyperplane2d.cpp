#include "hyperplane2d.hpp"

double Hyperplane2d::getBodyF(const xt::xtensor<double, 1>& P) const{
    return xt::linalg::vdot(a, P) + b + 1;
}

xt::xtensor<double, 1> Hyperplane2d::getBodyFdP(const xt::xtensor<double, 1>& P) const{
    return a;
}

xt::xtensor<double, 2> Hyperplane2d::getBodyFdPdP(const xt::xtensor<double, 1>& P) const{
    return xt::zeros<double>({2, 2});
}

xt::xtensor<double, 3> Hyperplane2d::getBodyFdPdPdP(const xt::xtensor<double, 1>& P) const{
    return xt::zeros<double>({2, 2, 2});
}