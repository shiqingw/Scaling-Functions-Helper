#include "superquadrics3d.hpp"
double signum(double n) {
    if (n > 0.0) {
        return 1.0;
    } else if (n < 0.0) {
        return -1.0;
    } else {
        return 0.0;
    }
}

double Superquadrics3d::getBodyF(const xt::xarray<double>& P) const{

    double a1 = a(0), a2 = a(1), a3 = a(2);
    double c1 = c(0), c2 = c(1), c3 = c(2);
    double x = std::abs(P(0)-c1)/a1, y = std::abs(P(1)-c2)/a2, z = std::abs(P(2)-c3)/a3;

    return std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), e2/e1) + std::pow(z, 2/e1);
}

xt::xarray<double> Superquadrics3d::getBodyFdP(const xt::xarray<double>& P) const{

    double a1 = a(0), a2 = a(1), a3 = a(2);
    double c1 = c(0), c2 = c(1), c3 = c(2);
    double x = std::abs(P(0)-c1)/a1, y = std::abs(P(1)-c2)/a2, z = std::abs(P(2)-c3)/a3;
    double sx = signum(P(0)-c1)/a1, sy = signum(P(1)-c2)/a2, sz = signum(P(2)-c3)/a3;

    xt::xarray<double> F_dP = xt::zeros<double>({3});
    F_dP(0) = 2*std::pow(x, (2 - e2)/e2)*std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), (-e1 + e2)/e1)/e1;
    F_dP(0) = sx * F_dP(0);
    F_dP(1) = 2*std::pow(y, (2 - e2)/e2)*std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), (-e1 + e2)/e1)/e1;
    F_dP(1) = sy * F_dP(1);
    F_dP(2) = 2*std::pow(z, (2 - e1)/e1)/e1;
    F_dP(2) = sz * F_dP(2);

    return F_dP;
}

xt::xarray<double> Superquadrics3d::getBodyFdPdP(const xt::xarray<double>& P) const{

    double a1 = a(0), a2 = a(1), a3 = a(2);
    double c1 = c(0), c2 = c(1), c3 = c(2);
    double x = std::abs(P(0)-c1)/a1, y = std::abs(P(1)-c2)/a2, z = std::abs(P(2)-c3)/a3;
    double sx = signum(P(0)-c1)/a1, sy = signum(P(1)-c2)/a2, sz = signum(P(2)-c3)/a3;

    xt::xarray<double> F_dPdP = xt::zeros<double>({3,3});
    F_dPdP(0,0) = 2*std::pow(x, -2 + 2/e2)*std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), e2/e1)*(-e1*e2*std::pow(x, 2/e2) - e1*e2*std::pow(y, 2/e2) + 2*e1*std::pow(y, 2/e2) + 2*e2*std::pow(x, 2/e2))/(std::pow(e1, 2)*e2*(std::pow(x, 4/e2) + 2*std::pow(x, 2/e2)*std::pow(y, 2/e2) + std::pow(y, 4/e2)));
    F_dPdP(0,0) = sx * sx * F_dPdP(0,0);
    F_dPdP(0,1) = 4*std::pow(x, -2 + (e2 + 2)/e2)*std::pow(y, -2 + (e2 + 2)/e2)*(-e1 + e2)*std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), -4 + (2*e1 + e2)/e1)/(std::pow(e1, 2)*e2);
    F_dPdP(0,1) = sx * sy * F_dPdP(0,1);
    F_dPdP(1,0) = 4*std::pow(x, -2 + (e2 + 2)/e2)*std::pow(y, -2 + (e2 + 2)/e2)*(-e1 + e2)*std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), -4 + (2*e1 + e2)/e1)/(std::pow(e1, 2)*e2);
    F_dPdP(1,0) = sy * sx * F_dPdP(1,0);
    F_dPdP(1,1) = 2*std::pow(y, -2 + 2/e2)*std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), e2/e1)*(-e1*e2*std::pow(x, 2/e2) - e1*e2*std::pow(y, 2/e2) + 2*e1*std::pow(x, 2/e2) + 2*e2*std::pow(y, 2/e2))/(std::pow(e1, 2)*e2*(std::pow(x, 4/e2) + 2*std::pow(x, 2/e2)*std::pow(y, 2/e2) + std::pow(y, 4/e2)));
    F_dPdP(1,1) = sy * sy * F_dPdP(1,1);
    F_dPdP(2,2) = 2*std::pow(z, -2 + 2/e1)*(2 - e1)/std::pow(e1, 2);
    F_dPdP(2,2) = sz * sz * F_dPdP(2,2);

    return F_dPdP;
}

xt::xarray<double> Superquadrics3d::getBodyFdPdPdP(const xt::xarray<double>& P) const{

    double a1 = a(0), a2 = a(1), a3 = a(2);
    double c1 = c(0), c2 = c(1), c3 = c(2);
    double x = std::abs(P(0)-c1)/a1, y = std::abs(P(1)-c2)/a2, z = std::abs(P(2)-c3)/a3;
    double sx = signum(P(0)-c1)/a1, sy = signum(P(1)-c2)/a2, sz = signum(P(2)-c3)/a3;

    xt::xarray<double> F_dPdPdP = xt::zeros<double>({3,3,3});
    F_dPdPdP(0,0,0) = 4*std::pow(x, -3 + 2/e2)*std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), e2/e1)*(std::pow(e1, 2)*std::pow(e2, 2)*std::pow(x, 4/e2) + 2*std::pow(e1, 2)*std::pow(e2, 2)*std::pow(x, 2/e2)*std::pow(y, 2/e2) + std::pow(e1, 2)*std::pow(e2, 2)*std::pow(y, 4/e2) - 3*std::pow(e1, 2)*e2*std::pow(x, 2/e2)*std::pow(y, 2/e2) - 3*std::pow(e1, 2)*e2*std::pow(y, 4/e2) - 2*std::pow(e1, 2)*std::pow(x, 2/e2)*std::pow(y, 2/e2) + 2*std::pow(e1, 2)*std::pow(y, 4/e2) - 3*e1*std::pow(e2, 2)*std::pow(x, 4/e2) - 3*e1*std::pow(e2, 2)*std::pow(x, 2/e2)*std::pow(y, 2/e2) + 6*e1*e2*std::pow(x, 2/e2)*std::pow(y, 2/e2) + 2*std::pow(e2, 2)*std::pow(x, 4/e2))/(std::pow(e1, 3)*std::pow(e2, 2)*(std::pow(x, 6/e2) + 3*std::pow(x, 4/e2)*std::pow(y, 2/e2) + 3*std::pow(x, 2/e2)*std::pow(y, 4/e2) + std::pow(y, 6/e2)));
    F_dPdPdP(0,0,0) = sx * sx * sx * F_dPdPdP(0,0,0);
    F_dPdPdP(0,0,1) = 4*std::pow(x, -2 + 2/e2)*std::pow(y, -1 + 2/e2)*std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), e2/e1)*(std::pow(e1, 2)*e2*std::pow(x, 2/e2) + std::pow(e1, 2)*e2*std::pow(y, 2/e2) + 2*std::pow(e1, 2)*std::pow(x, 2/e2) - 2*std::pow(e1, 2)*std::pow(y, 2/e2) - e1*std::pow(e2, 2)*std::pow(x, 2/e2) - e1*std::pow(e2, 2)*std::pow(y, 2/e2) - 4*e1*e2*std::pow(x, 2/e2) + 2*e1*e2*std::pow(y, 2/e2) + 2*std::pow(e2, 2)*std::pow(x, 2/e2))/(std::pow(e1, 3)*std::pow(e2, 2)*(std::pow(x, 6/e2) + 3*std::pow(x, 4/e2)*std::pow(y, 2/e2) + 3*std::pow(x, 2/e2)*std::pow(y, 4/e2) + std::pow(y, 6/e2)));
    F_dPdPdP(0,0,1) = sx * sx * sy * F_dPdPdP(0,0,1);
    F_dPdPdP(0,1,0) = 4*std::pow(x, -2 + 2/e2)*std::pow(y, -1 + 2/e2)*std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), e2/e1)*(std::pow(e1, 2)*e2*std::pow(x, 2/e2) + std::pow(e1, 2)*e2*std::pow(y, 2/e2) + 2*std::pow(e1, 2)*std::pow(x, 2/e2) - 2*std::pow(e1, 2)*std::pow(y, 2/e2) - e1*std::pow(e2, 2)*std::pow(x, 2/e2) - e1*std::pow(e2, 2)*std::pow(y, 2/e2) - 4*e1*e2*std::pow(x, 2/e2) + 2*e1*e2*std::pow(y, 2/e2) + 2*std::pow(e2, 2)*std::pow(x, 2/e2))/(std::pow(e1, 3)*std::pow(e2, 2)*(std::pow(x, 6/e2) + 3*std::pow(x, 4/e2)*std::pow(y, 2/e2) + 3*std::pow(x, 2/e2)*std::pow(y, 4/e2) + std::pow(y, 6/e2)));
    F_dPdPdP(0,1,0) = sx * sy * sx * F_dPdPdP(0,1,0);
    F_dPdPdP(0,1,1) = 4*std::pow(x, -1 + 2/e2)*std::pow(y, -2 + 2/e2)*std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), e2/e1)*(std::pow(e1, 2)*e2*std::pow(x, 2/e2) + std::pow(e1, 2)*e2*std::pow(y, 2/e2) - 2*std::pow(e1, 2)*std::pow(x, 2/e2) + 2*std::pow(e1, 2)*std::pow(y, 2/e2) - e1*std::pow(e2, 2)*std::pow(x, 2/e2) - e1*std::pow(e2, 2)*std::pow(y, 2/e2) + 2*e1*e2*std::pow(x, 2/e2) - 4*e1*e2*std::pow(y, 2/e2) + 2*std::pow(e2, 2)*std::pow(y, 2/e2))/(std::pow(e1, 3)*std::pow(e2, 2)*(std::pow(x, 6/e2) + 3*std::pow(x, 4/e2)*std::pow(y, 2/e2) + 3*std::pow(x, 2/e2)*std::pow(y, 4/e2) + std::pow(y, 6/e2)));
    F_dPdPdP(0,1,1) = sx * sy * sy * F_dPdPdP(0,1,1);
    F_dPdPdP(1,0,0) = 4*std::pow(x, -2 + 2/e2)*std::pow(y, -1 + 2/e2)*std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), e2/e1)*(std::pow(e1, 2)*e2*std::pow(x, 2/e2) + std::pow(e1, 2)*e2*std::pow(y, 2/e2) + 2*std::pow(e1, 2)*std::pow(x, 2/e2) - 2*std::pow(e1, 2)*std::pow(y, 2/e2) - e1*std::pow(e2, 2)*std::pow(x, 2/e2) - e1*std::pow(e2, 2)*std::pow(y, 2/e2) - 4*e1*e2*std::pow(x, 2/e2) + 2*e1*e2*std::pow(y, 2/e2) + 2*std::pow(e2, 2)*std::pow(x, 2/e2))/(std::pow(e1, 3)*std::pow(e2, 2)*(std::pow(x, 6/e2) + 3*std::pow(x, 4/e2)*std::pow(y, 2/e2) + 3*std::pow(x, 2/e2)*std::pow(y, 4/e2) + std::pow(y, 6/e2)));
    F_dPdPdP(1,0,0) = sy * sx * sx * F_dPdPdP(1,0,0);
    F_dPdPdP(1,0,1) = 4*std::pow(x, -1 + 2/e2)*std::pow(y, -2 + 2/e2)*std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), e2/e1)*(std::pow(e1, 2)*e2*std::pow(x, 2/e2) + std::pow(e1, 2)*e2*std::pow(y, 2/e2) - 2*std::pow(e1, 2)*std::pow(x, 2/e2) + 2*std::pow(e1, 2)*std::pow(y, 2/e2) - e1*std::pow(e2, 2)*std::pow(x, 2/e2) - e1*std::pow(e2, 2)*std::pow(y, 2/e2) + 2*e1*e2*std::pow(x, 2/e2) - 4*e1*e2*std::pow(y, 2/e2) + 2*std::pow(e2, 2)*std::pow(y, 2/e2))/(std::pow(e1, 3)*std::pow(e2, 2)*(std::pow(x, 6/e2) + 3*std::pow(x, 4/e2)*std::pow(y, 2/e2) + 3*std::pow(x, 2/e2)*std::pow(y, 4/e2) + std::pow(y, 6/e2)));
    F_dPdPdP(1,0,1) = sy * sx * sy * F_dPdPdP(1,0,1);
    F_dPdPdP(1,1,0) = 4*std::pow(x, -1 + 2/e2)*std::pow(y, -2 + 2/e2)*std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), e2/e1)*(std::pow(e1, 2)*e2*std::pow(x, 2/e2) + std::pow(e1, 2)*e2*std::pow(y, 2/e2) - 2*std::pow(e1, 2)*std::pow(x, 2/e2) + 2*std::pow(e1, 2)*std::pow(y, 2/e2) - e1*std::pow(e2, 2)*std::pow(x, 2/e2) - e1*std::pow(e2, 2)*std::pow(y, 2/e2) + 2*e1*e2*std::pow(x, 2/e2) - 4*e1*e2*std::pow(y, 2/e2) + 2*std::pow(e2, 2)*std::pow(y, 2/e2))/(std::pow(e1, 3)*std::pow(e2, 2)*(std::pow(x, 6/e2) + 3*std::pow(x, 4/e2)*std::pow(y, 2/e2) + 3*std::pow(x, 2/e2)*std::pow(y, 4/e2) + std::pow(y, 6/e2)));
    F_dPdPdP(1,1,0) = sy * sy * sx * F_dPdPdP(1,1,0);
    F_dPdPdP(1,1,1) = 4*std::pow(y, -3 + 2/e2)*std::pow(std::pow(x, 2/e2) + std::pow(y, 2/e2), e2/e1)*(std::pow(e1, 2)*std::pow(e2, 2)*std::pow(x, 4/e2) + 2*std::pow(e1, 2)*std::pow(e2, 2)*std::pow(x, 2/e2)*std::pow(y, 2/e2) + std::pow(e1, 2)*std::pow(e2, 2)*std::pow(y, 4/e2) - 3*std::pow(e1, 2)*e2*std::pow(x, 4/e2) - 3*std::pow(e1, 2)*e2*std::pow(x, 2/e2)*std::pow(y, 2/e2) + 2*std::pow(e1, 2)*std::pow(x, 4/e2) - 2*std::pow(e1, 2)*std::pow(x, 2/e2)*std::pow(y, 2/e2) - 3*e1*std::pow(e2, 2)*std::pow(x, 2/e2)*std::pow(y, 2/e2) - 3*e1*std::pow(e2, 2)*std::pow(y, 4/e2) + 6*e1*e2*std::pow(x, 2/e2)*std::pow(y, 2/e2) + 2*std::pow(e2, 2)*std::pow(y, 4/e2))/(std::pow(e1, 3)*std::pow(e2, 2)*(std::pow(x, 6/e2) + 3*std::pow(x, 4/e2)*std::pow(y, 2/e2) + 3*std::pow(x, 2/e2)*std::pow(y, 4/e2) + std::pow(y, 6/e2)));
    F_dPdPdP(1,1,1) = sy * sy * sy * F_dPdPdP(1,1,1);
    F_dPdPdP(2,2,2) = 4*std::pow(z, -3 + 2/e1)*(std::pow(e1, 2) - 3*e1 + 2)/std::pow(e1, 3);
    F_dPdPdP(2,2,2) = sz * sz * sz * F_dPdPdP(2,2,2);

    return F_dPdPdP;
}