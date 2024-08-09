#include "hyperplane3d.hpp"

double Hyperplane3d::getBodyF(const xt::xtensor<double, 1>& P) const {
    return xt::linalg::vdot(a, P) + b + 1;
}

xt::xtensor<double, 1> Hyperplane3d::getBodyFdP(const xt::xtensor<double, 1>& P) const {
    return a;
}

xt::xtensor<double, 2> Hyperplane3d::getBodyFdPdP(const xt::xtensor<double, 1>& P) const {
    return xt::zeros<double>({3, 3});
}

xt::xtensor<double, 3> Hyperplane3d::getBodyFdPdPdP(const xt::xtensor<double, 1>& P) const {
    return xt::zeros<double>({3, 3, 3});
}

xt::xtensor<double, 1> Hyperplane3d::getWorldSlope(const xt::xtensor<double, 1>& q) const {
    xt::xtensor<double, 2> R = getRotationMatrix(q);
    return xt::linalg::dot(R, a);
}

double Hyperplane3d::getWorldOffset(const xt::xtensor<double, 1>& d, const xt::xtensor<double, 1>& q) const {
    xt::xtensor<double, 2> R = getRotationMatrix(q);
    return b - xt::linalg::vdot(a, xt::linalg::dot(xt::transpose(R, {1,0}), d));
}