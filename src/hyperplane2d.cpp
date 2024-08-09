#include "hyperplane2d.hpp"

double Hyperplane2d::getBodyF(const xt::xtensor<double, 1>& P) const {
    return xt::linalg::vdot(a, P) + b + 1;
}

xt::xtensor<double, 1> Hyperplane2d::getBodyFdP(const xt::xtensor<double, 1>& P) const {
    return a;
}

xt::xtensor<double, 2> Hyperplane2d::getBodyFdPdP(const xt::xtensor<double, 1>& P) const {
    return xt::zeros<double>({2, 2});
}

xt::xtensor<double, 3> Hyperplane2d::getBodyFdPdPdP(const xt::xtensor<double, 1>& P) const {
    return xt::zeros<double>({2, 2, 2});
}

xt::xtensor<double, 1> Hyperplane2d::getWorldSlope(double theta) const {
    xt::xtensor<double, 2> R = getRotationMatrix(theta);
    return xt::linalg::dot(R, a);
}

double Hyperplane2d::getWorldOffset(const xt::xtensor<double, 1>& d, double theta) const {
    xt::xtensor<double, 2> R = getRotationMatrix(theta);
    return b - xt::linalg::vdot(a, xt::linalg::dot(xt::transpose(R, {1,0}), d));
}