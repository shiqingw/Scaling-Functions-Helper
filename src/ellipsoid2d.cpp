#include "ellipsoid2d.hpp"

double Ellipsoid2d::getBodyF(const xt::xtensor<double, 1>& P) const {
    return xt::linalg::dot(P-mu, xt::linalg::dot(Q, P-mu))(0);
}

xt::xtensor<double, 1> Ellipsoid2d::getBodyFdP(const xt::xtensor<double, 1>& P) const {
    return 2 * xt::linalg::dot(Q, P - mu);
}

xt::xtensor<double, 2> Ellipsoid2d::getBodyFdPdP(const xt::xtensor<double, 1>& P) const {
    return 2 * Q;
}

xt::xtensor<double, 3> Ellipsoid2d::getBodyFdPdPdP(const xt::xtensor<double, 1>& P) const {
    return xt::zeros<double>({2, 2, 2});
}

xt::xtensor<double, 2> Ellipsoid2d::getWorldQuadraticCoefficient(double theta) const {
    xt::xtensor<double, 2> R = getRotationMatrix(theta);
    return xt::linalg::dot(xt::linalg::dot(R, Q), xt::transpose(R, {1,0}));
}

xt::xtensor<double, 1> Ellipsoid2d::getWorldCenter(const xt::xtensor<double, 1>& d, double theta) const {
    xt::xtensor<double, 2> R = getRotationMatrix(theta);
    return xt::linalg::dot(R, mu) + d;
}