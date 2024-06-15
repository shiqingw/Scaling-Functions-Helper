#include "logSumExp3d.hpp"

double LogSumExp3d::getBodyF(const xt::xarray<double>& P) const{

    int dim_z = A.shape()[0];
    xt::xarray<double> z = kappa * (xt::linalg::dot(A, P) + b);
    double c = xt::amax(z)();
    z = xt::exp(z - c);
    double sum_z = xt::sum(z)();
    double F = log(sum_z) + c - log((double)dim_z) + 1;

    return F;
}

xt::xarray<double> LogSumExp3d::getBodyFdP(const xt::xarray<double>& P) const{
    
    int dim_p = P.shape()[0], dim_z = A.shape()[0];
    xt::xarray<double> z = kappa * (xt::linalg::dot(A, P) + b);
    double c = xt::amax(z)();
    z = xt::exp(z - c);
    double sum_z = xt::sum(z)();

    xt::xarray<double> zT_A = xt::linalg::dot(z, A); // shape: (dim_p,)
    xt::xarray<double> F_dp = kappa * zT_A / sum_z; // shape: (dim_p,)

    return F_dp; // shape: (dim_p,)
}

xt::xarray<double> LogSumExp3d::getBodyFdPdP(const xt::xarray<double>& P) const{
    
    int dim_p = P.shape()[0], dim_z = A.shape()[0];
    xt::xarray<double> z = kappa * (xt::linalg::dot(A, P) + b);
    double c = xt::amax(z)();
    z = xt::exp(z - c);
    double sum_z = xt::sum(z)();

    xt::xarray<double> zT_A = xt::linalg::dot(z, A); // shape: (dim_p,)

    xt::xarray<double> diag_z = xt::diag(z);
    xt::xarray<double> diag_z_A = xt::linalg::dot(diag_z, A);
    xt::xarray<double> AT_diag_z_A = xt::linalg::dot(xt::transpose(A), diag_z_A);
    xt::xarray<double> AT_z_zT_A = xt::linalg::outer(zT_A, zT_A);
    xt::xarray<double> F_dpdp = pow(kappa,2) * (AT_diag_z_A/sum_z - AT_z_zT_A/pow(sum_z,2));

    return F_dpdp; // shape: (dim_p, dim_p)
}

xt::xarray<double> LogSumExp3d::getBodyFdPdPdP(const xt::xarray<double>& P) const{
    
    int dim_p = P.shape()[0], dim_z = A.shape()[0];
    xt::xarray<double> z = kappa * (xt::linalg::dot(A, P) + b);
    double c = xt::amax(z)();
    z = xt::exp(z - c);
    double sum_z = xt::sum(z)();

    xt::xarray<double> zT_A = xt::linalg::dot(z, A); // shape: (dim_p,)

    xt::xarray<double> diag_z = xt::diag(z);
    xt::xarray<double> diag_z_A = xt::linalg::dot(diag_z, A);
    xt::xarray<double> AT_diag_z_A = xt::linalg::dot(xt::transpose(A), diag_z_A);
    xt::xarray<double> AT_z_zT_A = xt::linalg::outer(zT_A, zT_A);
    
    xt::xarray<double> F_dpdpdp = xt::zeros<double>({dim_p, dim_p, dim_p});
    xt::xarray<double> z_dp = kappa * diag_z_A;
    // part 1 of F_dpdpdp
    for (int i = 0; i < dim_p; ++i){
        xt::xarray<double> z_dp_i = xt::view(z_dp, xt::all(), i);
        xt::xarray<double> tmp = pow(kappa,2)*xt::linalg::dot(xt::transpose(A), xt::linalg::dot(xt::diag(z_dp_i) ,A))/sum_z;
        tmp -= pow(kappa,2) * xt::sum(z_dp_i)() * AT_diag_z_A/pow(sum_z,2);
        xt::view(F_dpdpdp, xt::all(), xt::all(), i) += tmp; 
    }
    // part 2 of F_dpdpdp
    xt::xarray<double> big = xt::zeros<double>({dim_z, dim_z, dim_z});
    xt::xarray<double> identity = xt::eye(dim_z);
    for (int i = 0; i < dim_z; ++i){
        xt::xarray<double> e_i = xt::view(identity, i, xt::all());
        xt::view(big, xt::all(), xt::all(), i) += xt::linalg::outer(e_i, z) + xt::linalg::outer(z, e_i);
    }

    for (int i = 0; i < dim_p; ++i){
        xt::xarray<double> z_dp_i = xt::view(z_dp, xt::all(), i);
        xt::xarray<double> tmp = xt::linalg::tensordot(big, z_dp_i, {2}, {0}); // shape dim_z x dim_z
        tmp = pow(kappa,2) * xt::linalg::dot(xt::transpose(A), xt::linalg::dot(tmp, A))/ pow(sum_z,2); // shape dim_p x dim_p
        tmp -= 2 * pow(kappa,2) * xt::sum(z_dp_i)() * AT_z_zT_A / pow(sum_z,3);
        xt::view(F_dpdpdp, xt::all(), xt::all(), i) -= tmp;
    }

    return F_dpdpdp;
}