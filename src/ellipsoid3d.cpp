#include "ellipsoid3d.hpp"

double Ellipsoid3d::getBodyF(const xt::xtensor<double, 1>& P) const{
    return xt::linalg::dot(P-mu, xt::linalg::dot(Q, P-mu))(0);
}

xt::xtensor<double, 1> Ellipsoid3d::getBodyFdP(const xt::xtensor<double, 1>& P) const{
    return 2 * xt::linalg::dot(Q, P - mu);
}

xt::xtensor<double, 2> Ellipsoid3d::getBodyFdPdP(const xt::xtensor<double, 1>& P) const{
    return 2 * Q;
}

xt::xtensor<double, 3> Ellipsoid3d::getBodyFdPdPdP(const xt::xtensor<double, 1>& P) const{
    return xt::zeros<double>({3, 3, 3});
}

xt::xtensor<double, 2> Ellipsoid3d::getWorldQuadraticCoefficient(const xt::xtensor<double, 1>& q) const{
    xt::xtensor<double, 2> R = getRotationMatrix(q);
    return xt::linalg::dot(xt::linalg::dot(R, Q), xt::transpose(R, {1,0}));
}

xt::xtensor<double, 1> Ellipsoid3d::getWorldCenter(const xt::xtensor<double, 1>& d, const xt::xtensor<double, 1>& q) const{
    xt::xtensor<double, 2> R = getRotationMatrix(q);
    return xt::linalg::dot(R, mu) + d;
}