#include "logSumExp3d.hpp"

double LogSumExp3d::getBodyF(const xt::xtensor<double, 1>& P) const {

    int dim_z = A.shape()[0];
    xt::xtensor<double, 1> z = kappa * (xt::linalg::dot(A, P) + b);
    double c = xt::amax(z)();
    z = xt::exp(z - c)/(double)dim_z;
    double sum_z = xt::sum(z)();
    double F = log(sum_z) + c + 1;

    return F;
}

xt::xtensor<double, 1> LogSumExp3d::getBodyFdP(const xt::xtensor<double, 1>& P) const {
    
    int dim_p = P.shape()[0], dim_z = A.shape()[0];
    xt::xtensor<double, 1> z = kappa * (xt::linalg::dot(A, P) + b);
    double c = xt::amax(z)();
    z = xt::exp(z - c)/(double)dim_z;
    double sum_z = xt::sum(z)();

    xt::xtensor<double, 1> zT_A = xt::linalg::dot(z, A); // shape: (dim_p,)
    xt::xtensor<double, 1> F_dp = kappa * zT_A / sum_z; // shape: (dim_p,)

    return F_dp; // shape: (dim_p,)
}

xt::xtensor<double, 2> LogSumExp3d::getBodyFdPdP(const xt::xtensor<double, 1>& P) const {
    
    int dim_p = P.shape()[0], dim_z = A.shape()[0];
    xt::xtensor<double, 1> z = kappa * (xt::linalg::dot(A, P) + b);
    double c = xt::amax(z)();
    z = xt::exp(z - c)/(double)dim_z;
    double sum_z = xt::sum(z)();

    xt::xtensor<double, 1> zT_A = xt::linalg::dot(z, A); // shape: (dim_p,)

    xt::xtensor<double, 2> diag_z = xt::diag(z);
    xt::xtensor<double, 2> diag_z_A = xt::linalg::dot(diag_z, A);
    xt::xtensor<double, 2> AT_diag_z_A = xt::linalg::dot(xt::transpose(A), diag_z_A);
    xt::xtensor<double, 2> AT_z_zT_A = xt::linalg::outer(zT_A, zT_A);
    xt::xtensor<double, 2> F_dpdp = pow(kappa,2) * (AT_diag_z_A/sum_z - AT_z_zT_A/pow(sum_z,2));

    return F_dpdp; // shape: (dim_p, dim_p)
}

xt::xtensor<double, 3> LogSumExp3d::getBodyFdPdPdP(const xt::xtensor<double, 1>& P) const {
    
    int dim_p = P.shape()[0], dim_z = A.shape()[0];
    xt::xtensor<double, 1> z = kappa * (xt::linalg::dot(A, P) + b);
    double c = xt::amax(z)();
    z = xt::exp(z - c)/(double)dim_z;
    double sum_z = xt::sum(z)();

    xt::xtensor<double, 1> zT_A = xt::linalg::dot(z, A); // shape: (dim_p,)

    xt::xtensor<double, 2> diag_z = xt::diag(z);
    xt::xtensor<double, 2> diag_z_A = xt::linalg::dot(diag_z, A);
    xt::xtensor<double, 2> AT_diag_z_A = xt::linalg::dot(xt::transpose(A), diag_z_A);
    xt::xtensor<double, 2> AT_z_zT_A = xt::linalg::outer(zT_A, zT_A);
    
    xt::xtensor<double, 3> F_dpdpdp = xt::zeros<double>({dim_p, dim_p, dim_p});
    xt::xtensor<double, 2> z_dp = kappa * diag_z_A;
    // part 1 of F_dpdpdp
    for (int i = 0; i < dim_p; ++i){
        xt::xtensor<double, 1> z_dp_i = xt::view(z_dp, xt::all(), i);
        xt::xtensor<double, 2> tmp = pow(kappa,2)*xt::linalg::dot(xt::transpose(A), xt::linalg::dot(xt::diag(z_dp_i) ,A))/sum_z;
        tmp -= pow(kappa,2) * xt::sum(z_dp_i)() * AT_diag_z_A/pow(sum_z,2);
        xt::view(F_dpdpdp, xt::all(), xt::all(), i) += tmp; 
    }
    // part 2 of F_dpdpdp
    xt::xtensor<double, 3> big = xt::zeros<double>({dim_z, dim_z, dim_z});
    xt::xtensor<double, 2> identity = xt::eye(dim_z);
    for (int i = 0; i < dim_z; ++i){
        xt::xtensor<double, 1> e_i = xt::view(identity, i, xt::all());
        xt::view(big, xt::all(), xt::all(), i) += xt::linalg::outer(e_i, z) + xt::linalg::outer(z, e_i);
    }

    for (int i = 0; i < dim_p; ++i){
        xt::xtensor<double, 1> z_dp_i = xt::view(z_dp, xt::all(), i);
        xt::xtensor<double, 2> tmp = xt::linalg::tensordot(big, z_dp_i, {2}, {0}); // shape dim_z x dim_z
        tmp = pow(kappa,2) * xt::linalg::dot(xt::transpose(A), xt::linalg::dot(tmp, A))/ pow(sum_z,2); // shape dim_p x dim_p
        tmp -= 2 * pow(kappa,2) * xt::sum(z_dp_i)() * AT_z_zT_A / pow(sum_z,3);
        xt::view(F_dpdpdp, xt::all(), xt::all(), i) -= tmp;
    }

    return F_dpdpdp;
}

xt::xtensor<double, 2> LogSumExp3d::getWorldMatrixCoefficient(const xt::xtensor<double, 1>& q) const {

    xt::xtensor<double, 2> R = getRotationMatrix(q);
    return xt::linalg::dot(A, xt::transpose(R, {1,0}));
}

xt::xtensor<double, 1> LogSumExp3d::getWorldVectorCoefficient(const xt::xtensor<double, 1>& d, 
    const xt::xtensor<double, 1>& q) const {
    
    xt::xtensor<double, 2> R = getRotationMatrix(q);
    return -xt::linalg::dot(A, xt::linalg::dot(xt::transpose(R, {1,0}), d)) + b;
}

