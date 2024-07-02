#include "diffOptHelper.hpp"

double getDualVariable(const xt::xarray<double>& F1_dp, const xt::xarray<double>& F2_dp){

    double F1_dp_norm = xt::linalg::norm(F1_dp, 2);
    double F2_dp_norm = xt::linalg::norm(F2_dp, 2);

    return F1_dp_norm/F2_dp_norm;
}

xt::xarray<double> getGradientGeneral(double dual_var,
    const xt::xarray<double>& F1_dp,const xt::xarray<double>& F2_dp,
    const xt::xarray<double>& F1_dx, const xt::xarray<double>& F2_dx,
    const xt::xarray<double>& F1_dpdp, const xt::xarray<double>& F2_dpdp,
    const xt::xarray<double>& F1_dpdx, const xt::xarray<double>& F2_dpdx){

    int dim_p = xt::adapt(F1_dpdx.shape())(0);
    xt::xarray<double> b1 = - F1_dpdx - dual_var * F2_dpdx; // shape (dim_p, dim_x)
    xt::xarray<double> b2 = xt::view(-F2_dx, xt::newaxis(), xt::all()); // shape (1, dim_x)
    xt::xarray<double> b = xt::concatenate(xt::xtuple(b1, b2), 0); // shape (dim_p+1, dim_x)

    xt::xarray<double> A11 = F1_dpdp + dual_var * F2_dpdp; // shape (dim_p, dim_p)
    xt::xarray<double> A = xt::zeros<double>({dim_p+1, dim_p+1});
    xt::view(A, xt::range(0, dim_p), xt::range(0, dim_p)) = A11;
    xt::view(A, xt::range(0, dim_p), dim_p) = F2_dp;
    xt::view(A, dim_p, xt::range(0, dim_p)) = F2_dp;

    xt::xarray<double> grad = xt::linalg::solve(A, b); // shape (dim_p+1, dim_x)
    xt::xarray<double> p_dx = xt::view(grad, xt::range(0, dim_p), xt::all());
    xt::xarray<double> alpha_dx = F1_dx + xt::linalg::dot(F1_dp, p_dx); // shape (dim_x)

    return alpha_dx;
}

std::tuple<xt::xarray<double>,xt::xarray<double>> getGradientAndHessianGeneral(double dual_var,
    const xt::xarray<double>& F1_dp, const xt::xarray<double>& F2_dp,
    const xt::xarray<double>& F1_dx, const xt::xarray<double>& F2_dx,
    const xt::xarray<double>& F1_dpdp, const xt::xarray<double>& F2_dpdp,
    const xt::xarray<double>& F1_dpdx, const xt::xarray<double>& F2_dpdx,
    const xt::xarray<double>& F1_dxdx, const xt::xarray<double>& F2_dxdx,
    const xt::xarray<double>& F1_dpdpdp, const xt::xarray<double>& F2_dpdpdp,
    const xt::xarray<double>& F1_dpdpdx, const xt::xarray<double>& F2_dpdpdx,
    const xt::xarray<double>& F1_dpdxdx, const xt::xarray<double>& F2_dpdxdx){
    
    int dim_p = xt::adapt(F1_dpdx.shape())(0);
    int dim_x = xt::adapt(F1_dpdx.shape())(1);
    xt::xarray<double> b1 = - F1_dpdx - dual_var * F2_dpdx; // shape (dim_p, dim_x)
    xt::xarray<double> b2 = xt::view(-F2_dx, xt::newaxis(), xt::all()); // shape (1, dim_x)
    xt::xarray<double> b = xt::concatenate(xt::xtuple(b1, b2), 0); // shape (dim_p+1, dim_x)

    xt::xarray<double> A11 = F1_dpdp + dual_var * F2_dpdp; // shape (dim_p, dim_p)
    xt::xarray<double> A = xt::zeros<double>({dim_p+1, dim_p+1});
    xt::view(A, xt::range(0, dim_p), xt::range(0, dim_p)) = A11;
    xt::view(A, xt::range(0, dim_p), dim_p) = F2_dp;
    xt::view(A, dim_p, xt::range(0, dim_p)) = F2_dp;

    xt::xarray<double> grad = xt::linalg::solve(A, b); // shape (dim_p+1, dim_x)
    xt::xarray<double> p_dx = xt::view(grad, xt::range(0, dim_p), xt::all()); // shape (dim_p, dim_x)
    xt::xarray<double> dual_dx = xt::view(grad, dim_p, xt::all()); // shape (dim_x)
    xt::xarray<double> alpha_dx = F1_dx + xt::linalg::dot(F1_dp, p_dx); // shape (dim_x)

    xt::xarray<double> F1_dpdxdp = xt::transpose(F1_dpdpdx, {0, 2, 1}); // shape (dim_p, dim_x, dim_p)
    xt::xarray<double> F2_dpdxdp = xt::transpose(F2_dpdpdx, {0, 2, 1}); // shape (dim_p, dim_x, dim_p)
    xt::xarray<double> b1_dx_1 = -F1_dpdxdx - xt::linalg::tensordot(F1_dpdxdp, p_dx, {2}, {0}); // shape (dim_p, dim_x, dim_x)
    xt::xarray<double> tmp1 = F2_dpdxdx + xt::linalg::tensordot(F2_dpdxdp, p_dx, {2}, {0}); // shape (dim_p, dim_x, dim_x)
    xt::xarray<double> b1_dx_2 = - xt::linalg::tensordot(xt::view(F2_dpdx, xt::all(), xt::all(), xt::newaxis()),
                                xt::view(dual_dx, xt::newaxis(), xt::all()), {2}, {0}) - dual_var*tmp1; // shape (dim_p, dim_x, dim_x)
    xt::xarray<double> b1_dx = b1_dx_1 + b1_dx_2; // shape (dim_p, dim_x, dim_x)
    xt::xarray<double> F2_dxdp = xt::transpose(F2_dpdx); // shape (dim_x, dim_p)
    xt::xarray<double> b2_dx = -F2_dxdx - xt::linalg::dot(F2_dxdp, p_dx); // shape (dim_x, dim_x)
    xt::xarray<double> b_dx = xt::concatenate(xt::xtuple(b1_dx, xt::view(b2_dx, xt::newaxis(), xt::all(), xt::all())), 0); // shape (dim_p+1, dim_x, dim_x)

    xt::xarray<double> A11_dx_1 = F1_dpdpdx + xt::linalg::tensordot(F1_dpdpdp, p_dx, {2}, {0}); // shape (dim_p, dim_p, dim_x)
    xt::xarray<double> tmp2 = F2_dpdpdx + xt::linalg::tensordot(F2_dpdpdp, p_dx, {2}, {0}); // shape (dim_p, dim_p, dim_x)
    xt::xarray<double> A11_dx_2 = xt::linalg::tensordot(xt::view(F2_dpdp, xt::all(), xt::all(), xt::newaxis()),
                                xt::view(dual_dx, xt::newaxis(), xt::all()), {2}, {0}) + dual_var*tmp2; // shape (dim_p, dim_p, dim_x)
    xt::xarray<double> A11_dx = A11_dx_1 + A11_dx_2; // shape (dim_p, dim_p, dim_x)
    xt::xarray<double> A21_dx = F2_dpdx + xt::linalg::dot(F2_dpdp, p_dx); // shape (dim_p, dim_x)
    xt::xarray<double> A_dx = xt::zeros<double>({dim_p+1, dim_p+1, dim_x});
    xt::view(A_dx, xt::range(0, dim_p), xt::range(0, dim_p), xt::all()) = A11_dx;
    xt::view(A_dx, dim_p, xt::range(0, dim_p), xt::all()) = A21_dx;
    xt::view(A_dx, xt::range(0, dim_p), dim_p, xt::all()) = A21_dx;

    xt::xarray<double> hessian = xt::zeros<double>({dim_p+1, dim_x, dim_x});
    for (int i = 0; i < dim_x; ++i){
        xt::xarray<double> A_dx_i = xt::view(A_dx, xt::all(), xt::all(), i); // shape (dim_p+1, dim_p+1)
        xt::xarray<double> b_dx_i = xt::view(b_dx, xt::all(), xt::all(), i); // shape (dim_p+1, dim_x)
        xt::view(hessian, xt::all(), xt::all(), i) = - xt::linalg::solve(A, xt::linalg::dot(A_dx_i, grad)) + xt::linalg::solve(A, b_dx_i);
    }
    xt::xarray<double> F1_dxdp = xt::transpose(F1_dpdx); // shape (dim_x, dim_p)
    xt::xarray<double> p_dxdx = xt::view(hessian, xt::range(0, dim_p), xt::all(), xt::all()); // shape (dim_p, dim_x, dim_x)
    xt::xarray<double> alpha_dxdx = F1_dxdx + xt::linalg::dot(F1_dxdp, p_dx) + xt::linalg::dot(xt::transpose(p_dx), F1_dpdx); // shape (dim_x, dim_x)
    alpha_dxdx += xt::linalg::dot(xt::transpose(p_dx), xt::linalg::dot(F1_dpdp, p_dx))
                    + xt::linalg::tensordot(F1_dp, p_dxdx, {0}, {0}); // shape (dim_x, dim_x)
    
    return std::make_tuple(alpha_dx, alpha_dxdx);
}

std::tuple<double, xt::xarray<double>> getGradient2dOld(
    const xt::xarray<double>& p, std::shared_ptr<ScalingFunction2d> SF1, const xt::xarray<double>& d1, double theta1,
    std::shared_ptr<ScalingFunction2d> SF2, const xt::xarray<double>& d2, double theta2){

    if (SF1->isMoving == false && SF2->isMoving == false){
        throw std::invalid_argument("Both scaling functions are not moving.");
    } else if (SF1->isMoving == true && SF2->isMoving == true){
        int dim_p = 2, dim_x1 = 3, dim_x2 = 3, dim_x = dim_x1 + dim_x2; // x = [x1,x2]
        double alpha = SF1->getWorldF(p, d1, theta1);

        xt::xarray<double> F1_dp = SF1->getWorldFdp(p, d1, theta1);
        xt::xarray<double> F1_dx1 = SF1->getWorldFdx(p, d1, theta1);
        xt::xarray<double> F1_dpdp = SF1->getWorldFdpdp(p, d1, theta1);
        xt::xarray<double> F1_dpdx1 = SF1->getWorldFdpdx(p, d1, theta1);

        xt::xarray<double> F1_dx = xt::zeros<double>({dim_x});
        xt::view(F1_dx, xt::range(0, dim_x1)) = F1_dx1;
        xt::xarray<double> F1_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F1_dpdx, xt::all(), xt::range(0, dim_x1)) = F1_dpdx1;

        xt::xarray<double> F2_dp = SF2->getWorldFdp(p, d2, theta2);
        xt::xarray<double> F2_dx2 = SF2->getWorldFdx(p, d2, theta2);
        xt::xarray<double> F2_dpdp = SF2->getWorldFdpdp(p, d2, theta2);
        xt::xarray<double> F2_dpdx2 = SF2->getWorldFdpdx(p, d2, theta2);

        xt::xarray<double> F2_dx = xt::zeros<double>({dim_x});
        xt::view(F2_dx, xt::range(dim_x1, dim_x)) = F2_dx2;
        xt::xarray<double> F2_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F2_dpdx, xt::all(), xt::range(dim_x1, dim_x)) = F2_dpdx2;
        xt::xarray<double> F2_dxdx = xt::zeros<double>({dim_x, dim_x});

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx = getGradientGeneral(dual_var, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx);

        return std::make_tuple(alpha, alpha_dx);
        
    } else { // one of the scaling functions is moving
        // int dim_p = 2, dim_x = 3;
        double alpha = SF1->getWorldF(p, d1, theta1);

        xt::xarray<double> F1_dp = SF1->getWorldFdp(p, d1, theta1);
        xt::xarray<double> F1_dx = SF1->getWorldFdx(p, d1, theta1);
        xt::xarray<double> F1_dpdp = SF1->getWorldFdpdp(p, d1, theta1);
        xt::xarray<double> F1_dpdx = SF1->getWorldFdpdx(p, d1, theta1);

        xt::xarray<double> F2_dp = SF2->getWorldFdp(p, d2, theta2);
        xt::xarray<double> F2_dx = SF2->getWorldFdx(p, d2, theta2);
        xt::xarray<double> F2_dpdp = SF2->getWorldFdpdp(p, d2, theta2);
        xt::xarray<double> F2_dpdx = SF2->getWorldFdpdx(p, d2, theta2);

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx = getGradientGeneral(dual_var, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx);

        return std::make_tuple(alpha, alpha_dx);
    }
}

std::tuple<double, xt::xarray<double>> getGradient3dOld(
    const xt::xarray<double>& p, std::shared_ptr<ScalingFunction3d> SF1, const xt::xarray<double>& d1, const xt::xarray<double>& q1,
    std::shared_ptr<ScalingFunction3d> SF2, const xt::xarray<double>& d2, const xt::xarray<double>& q2){

    if (SF1->isMoving == false && SF2->isMoving == false){
        throw std::invalid_argument("Both scaling functions are not moving.");
    } else if (SF1->isMoving == true && SF2->isMoving == true){
        int dim_p = 3, dim_x1 = 7, dim_x2 = 7, dim_x = dim_x1 + dim_x2; // x = [x1,x2]
        double alpha = SF1->getWorldF(p, d1, q1);

        xt::xarray<double> F1_dp = SF1->getWorldFdp(p, d1, q1);
        xt::xarray<double> F1_dx1 = SF1->getWorldFdx(p, d1, q1);
        xt::xarray<double> F1_dpdp = SF1->getWorldFdpdp(p, d1, q1);
        xt::xarray<double> F1_dpdx1 = SF1->getWorldFdpdx(p, d1, q1);

        xt::xarray<double> F1_dx = xt::zeros<double>({dim_x});
        xt::view(F1_dx, xt::range(0, dim_x1)) = F1_dx1;
        xt::xarray<double> F1_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F1_dpdx, xt::all(), xt::range(0, dim_x1)) = F1_dpdx1;

        xt::xarray<double> F2_dp = SF2->getWorldFdp(p, d2, q2);
        xt::xarray<double> F2_dx2 = SF2->getWorldFdx(p, d2, q2);
        xt::xarray<double> F2_dpdp = SF2->getWorldFdpdp(p, d2, q2);
        xt::xarray<double> F2_dpdx2 = SF2->getWorldFdpdx(p, d2, q2);

        xt::xarray<double> F2_dx = xt::zeros<double>({dim_x});
        xt::view(F2_dx, xt::range(dim_x1, dim_x)) = F2_dx2;
        xt::xarray<double> F2_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F2_dpdx, xt::all(), xt::range(dim_x1, dim_x)) = F2_dpdx2;
        xt::xarray<double> F2_dxdx = xt::zeros<double>({dim_x, dim_x});

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx = getGradientGeneral(dual_var, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx);

        return std::make_tuple(alpha, alpha_dx);
        
    } else { // one of the scaling functions is moving
        // int dim_p = 3, dim_x = 7;
        double alpha = SF1->getWorldF(p, d1, q1);

        xt::xarray<double> F1_dp = SF1->getWorldFdp(p, d1, q1);
        xt::xarray<double> F1_dx = SF1->getWorldFdx(p, d1, q1);
        xt::xarray<double> F1_dpdp = SF1->getWorldFdpdp(p, d1, q1);
        xt::xarray<double> F1_dpdx = SF1->getWorldFdpdx(p, d1, q1);

        xt::xarray<double> F2_dp = SF2->getWorldFdp(p, d2, q2);
        xt::xarray<double> F2_dx = SF2->getWorldFdx(p, d2, q2);
        xt::xarray<double> F2_dpdp = SF2->getWorldFdpdp(p, d2, q2);
        xt::xarray<double> F2_dpdx = SF2->getWorldFdpdx(p, d2, q2);

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx = getGradientGeneral(dual_var, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx);

        return std::make_tuple(alpha, alpha_dx);
    }
}

std::tuple<double, xt::xarray<double>, xt::xarray<double>> getGradientAndHessian2dOld(
    const xt::xarray<double>& p, std::shared_ptr<ScalingFunction2d> SF1, const xt::xarray<double>& d1, double theta1,
    std::shared_ptr<ScalingFunction2d> SF2, const xt::xarray<double>& d2, double theta2){

    if (SF1->isMoving == false && SF2->isMoving == false){
        throw std::invalid_argument("Both scaling functions are not moving.");
    } else if (SF1->isMoving == true && SF2->isMoving == true){
        int dim_p = 2, dim_x1 = 3, dim_x2 = 3, dim_x = dim_x1 + dim_x2; // x = [x1,x2]
        double alpha = SF1->getWorldF(p, d1, theta1);

        xt::xarray<double> F1_dp = SF1->getWorldFdp(p, d1, theta1);
        xt::xarray<double> F1_dx1 = SF1->getWorldFdx(p, d1, theta1);
        xt::xarray<double> F1_dpdp = SF1->getWorldFdpdp(p, d1, theta1);
        xt::xarray<double> F1_dpdx1 = SF1->getWorldFdpdx(p, d1, theta1);
        xt::xarray<double> F1_dx1dx1 = SF1->getWorldFdxdx(p, d1, theta1);
        xt::xarray<double> F1_dpdpdp = SF1->getWorldFdpdpdp(p, d1, theta1);
        xt::xarray<double> F1_dpdpdx1 = SF1->getWorldFdpdpdx(p, d1, theta1);
        xt::xarray<double> F1_dpdx1dx1 = SF1->getWorldFdpdxdx(p, d1, theta1);

        xt::xarray<double> F1_dx = xt::zeros<double>({dim_x});
        xt::view(F1_dx, xt::range(0, dim_x1)) = F1_dx1;
        xt::xarray<double> F1_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F1_dpdx, xt::all(), xt::range(0, dim_x1)) = F1_dpdx1;
        xt::xarray<double> F1_dxdx = xt::zeros<double>({dim_x, dim_x});
        xt::view(F1_dxdx, xt::range(0, dim_x1), xt::range(0, dim_x1)) = F1_dx1dx1;
        xt::xarray<double> F1_dpdpdx = xt::zeros<double>({dim_p, dim_p, dim_x});
        xt::view(F1_dpdpdx, xt::all(), xt::all(), xt::range(0, dim_x1)) = F1_dpdpdx1;
        xt::xarray<double> F1_dpdxdx = xt::zeros<double>({dim_p, dim_x, dim_x});
        xt::view(F1_dpdxdx, xt::all(), xt::range(0, dim_x1), xt::range(0, dim_x1)) = F1_dpdx1dx1;

        xt::xarray<double> F2_dp = SF2->getWorldFdp(p, d2, theta2);
        xt::xarray<double> F2_dx2 = SF2->getWorldFdx(p, d2, theta2);
        xt::xarray<double> F2_dpdp = SF2->getWorldFdpdp(p, d2, theta2);
        xt::xarray<double> F2_dpdx2 = SF2->getWorldFdpdx(p, d2, theta2);
        xt::xarray<double> F2_dx2dx2 = SF2->getWorldFdxdx(p, d2, theta2);
        xt::xarray<double> F2_dpdpdp = SF2->getWorldFdpdpdp(p, d2, theta2);
        xt::xarray<double> F2_dpdpdx2 = SF2->getWorldFdpdpdx(p, d2, theta2);
        xt::xarray<double> F2_dpdx2dx2 = SF2->getWorldFdpdxdx(p, d2, theta2);

        xt::xarray<double> F2_dx = xt::zeros<double>({dim_x});
        xt::view(F2_dx, xt::range(dim_x1, dim_x)) = F2_dx2;
        xt::xarray<double> F2_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F2_dpdx, xt::all(), xt::range(dim_x1, dim_x)) = F2_dpdx2;
        xt::xarray<double> F2_dxdx = xt::zeros<double>({dim_x, dim_x});
        xt::view(F2_dxdx, xt::range(dim_x1, dim_x), xt::range(dim_x1, dim_x)) = F2_dx2dx2;
        xt::xarray<double> F2_dpdpdx = xt::zeros<double>({dim_p, dim_p, dim_x});
        xt::view(F2_dpdpdx, xt::all(), xt::all(), xt::range(dim_x1, dim_x)) = F2_dpdpdx2;
        xt::xarray<double> F2_dpdxdx = xt::zeros<double>({dim_p, dim_x, dim_x});
        xt::view(F2_dpdxdx, xt::all(), xt::range(dim_x1, dim_x), xt::range(dim_x1, dim_x)) = F2_dpdx2dx2;

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx, alpha_dxdx;
        std::tie(alpha_dx, alpha_dxdx) = getGradientAndHessianGeneral(dual_var, 
            F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx, F1_dxdx, F2_dxdx,
            F1_dpdpdp, F2_dpdpdp, F1_dpdpdx, F2_dpdpdx, F1_dpdxdx, F2_dpdxdx);
        
        return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
        
    } else { // one of the scaling functions is moving
        // int dim_p = 2, dim_x = 3;
        double alpha = SF1->getWorldF(p, d1, theta1);

        xt::xarray<double> F1_dp = SF1->getWorldFdp(p, d1, theta1);
        xt::xarray<double> F1_dx = SF1->getWorldFdx(p, d1, theta1);
        xt::xarray<double> F1_dpdp = SF1->getWorldFdpdp(p, d1, theta1);
        xt::xarray<double> F1_dpdx = SF1->getWorldFdpdx(p, d1, theta1);
        xt::xarray<double> F1_dxdx = SF1->getWorldFdxdx(p, d1, theta1);
        xt::xarray<double> F1_dpdpdp = SF1->getWorldFdpdpdp(p, d1, theta1);
        xt::xarray<double> F1_dpdpdx = SF1->getWorldFdpdpdx(p, d1, theta1);
        xt::xarray<double> F1_dpdxdx = SF1->getWorldFdpdxdx(p, d1, theta1);

        xt::xarray<double> F2_dp = SF2->getWorldFdp(p, d2, theta2);
        xt::xarray<double> F2_dx = SF2->getWorldFdx(p, d2, theta2);
        xt::xarray<double> F2_dpdp = SF2->getWorldFdpdp(p, d2, theta2);
        xt::xarray<double> F2_dpdx = SF2->getWorldFdpdx(p, d2, theta2);
        xt::xarray<double> F2_dxdx = SF2->getWorldFdxdx(p, d2, theta2);
        xt::xarray<double> F2_dpdpdp = SF2->getWorldFdpdpdp(p, d2, theta2);
        xt::xarray<double> F2_dpdpdx = SF2->getWorldFdpdpdx(p, d2, theta2);
        xt::xarray<double> F2_dpdxdx = SF2->getWorldFdpdxdx(p, d2, theta2);

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx, alpha_dxdx;
        std::tie(alpha_dx, alpha_dxdx) = getGradientAndHessianGeneral(dual_var, 
            F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx, F1_dxdx, F2_dxdx,
            F1_dpdpdp, F2_dpdpdp, F1_dpdpdx, F2_dpdpdx, F1_dpdxdx, F2_dpdxdx);

        return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
    }
}

std::tuple<double, xt::xarray<double>, xt::xarray<double>> getGradientAndHessian3dOld(
    const xt::xarray<double>& p, std::shared_ptr<ScalingFunction3d> SF1, const xt::xarray<double>& d1, const xt::xarray<double>& q1,
    std::shared_ptr<ScalingFunction3d> SF2, const xt::xarray<double>& d2, const xt::xarray<double>& q2){

    if (SF1->isMoving == false && SF2->isMoving == false){
        throw std::invalid_argument("Both scaling functions are not moving.");
    } else if (SF1->isMoving == true && SF2->isMoving == true){
        int dim_p = 3, dim_x1 = 7, dim_x2 = 7, dim_x = dim_x1 + dim_x2; // x = [x1,x2]
        double alpha = SF1->getWorldF(p, d1, q1);

        xt::xarray<double> F1_dp = SF1->getWorldFdp(p, d1, q1);
        xt::xarray<double> F1_dx1 = SF1->getWorldFdx(p, d1, q1);
        xt::xarray<double> F1_dpdp = SF1->getWorldFdpdp(p, d1, q1);
        xt::xarray<double> F1_dpdx1 = SF1->getWorldFdpdx(p, d1, q1);
        xt::xarray<double> F1_dx1dx1 = SF1->getWorldFdxdx(p, d1, q1);
        xt::xarray<double> F1_dpdpdp = SF1->getWorldFdpdpdp(p, d1, q1);
        xt::xarray<double> F1_dpdpdx1 = SF1->getWorldFdpdpdx(p, d1, q1);
        xt::xarray<double> F1_dpdx1dx1 = SF1->getWorldFdpdxdx(p, d1, q1);

        xt::xarray<double> F1_dx = xt::zeros<double>({dim_x});
        xt::view(F1_dx, xt::range(0, dim_x1)) = F1_dx1;
        xt::xarray<double> F1_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F1_dpdx, xt::all(), xt::range(0, dim_x1)) = F1_dpdx1;
        xt::xarray<double> F1_dxdx = xt::zeros<double>({dim_x, dim_x});
        xt::view(F1_dxdx, xt::range(0, dim_x1), xt::range(0, dim_x1)) = F1_dx1dx1;
        xt::xarray<double> F1_dpdpdx = xt::zeros<double>({dim_p, dim_p, dim_x});
        xt::view(F1_dpdpdx, xt::all(), xt::all(), xt::range(0, dim_x1)) = F1_dpdpdx1;
        xt::xarray<double> F1_dpdxdx = xt::zeros<double>({dim_p, dim_x, dim_x});
        xt::view(F1_dpdxdx, xt::all(), xt::range(0, dim_x1), xt::range(0, dim_x1)) = F1_dpdx1dx1;

        xt::xarray<double> F2_dp = SF2->getWorldFdp(p, d2, q2);
        xt::xarray<double> F2_dx2 = SF2->getWorldFdx(p, d2, q2);
        xt::xarray<double> F2_dpdp = SF2->getWorldFdpdp(p, d2, q2);
        xt::xarray<double> F2_dpdx2 = SF2->getWorldFdpdx(p, d2, q2);
        xt::xarray<double> F2_dx2dx2 = SF2->getWorldFdxdx(p, d2, q2);
        xt::xarray<double> F2_dpdpdp = SF2->getWorldFdpdpdp(p, d2, q2);
        xt::xarray<double> F2_dpdpdx2 = SF2->getWorldFdpdpdx(p, d2, q2);
        xt::xarray<double> F2_dpdx2dx2 = SF2->getWorldFdpdxdx(p, d2, q2);

        xt::xarray<double> F2_dx = xt::zeros<double>({dim_x});
        xt::view(F2_dx, xt::range(dim_x1, dim_x)) = F2_dx2;
        xt::xarray<double> F2_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F2_dpdx, xt::all(), xt::range(dim_x1, dim_x)) = F2_dpdx2;
        xt::xarray<double> F2_dxdx = xt::zeros<double>({dim_x, dim_x});
        xt::view(F2_dxdx, xt::range(dim_x1, dim_x), xt::range(dim_x1, dim_x)) = F2_dx2dx2;
        xt::xarray<double> F2_dpdpdx = xt::zeros<double>({dim_p, dim_p, dim_x});
        xt::view(F2_dpdpdx, xt::all(), xt::all(), xt::range(dim_x1, dim_x)) = F2_dpdpdx2;
        xt::xarray<double> F2_dpdxdx = xt::zeros<double>({dim_p, dim_x, dim_x});
        xt::view(F2_dpdxdx, xt::all(), xt::range(dim_x1, dim_x), xt::range(dim_x1, dim_x)) = F2_dpdx2dx2;

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx, alpha_dxdx;
        std::tie(alpha_dx, alpha_dxdx) = getGradientAndHessianGeneral(dual_var, 
            F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx, F1_dxdx, F2_dxdx,
            F1_dpdpdp, F2_dpdpdp, F1_dpdpdx, F2_dpdpdx, F1_dpdxdx, F2_dpdxdx);
        
        return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
        
    } else { // one of the scaling functions is moving
        // int dim_p = 3, dim_x = 7;
        double alpha = SF1->getWorldF(p, d1, q1);

        xt::xarray<double> F1_dp = SF1->getWorldFdp(p, d1, q1);
        xt::xarray<double> F1_dx = SF1->getWorldFdx(p, d1, q1);
        xt::xarray<double> F1_dpdp = SF1->getWorldFdpdp(p, d1, q1);
        xt::xarray<double> F1_dpdx = SF1->getWorldFdpdx(p, d1, q1);
        xt::xarray<double> F1_dxdx = SF1->getWorldFdxdx(p, d1, q1);
        xt::xarray<double> F1_dpdpdp = SF1->getWorldFdpdpdp(p, d1, q1);
        xt::xarray<double> F1_dpdpdx = SF1->getWorldFdpdpdx(p, d1, q1);
        xt::xarray<double> F1_dpdxdx = SF1->getWorldFdpdxdx(p, d1, q1);

        xt::xarray<double> F2_dp = SF2->getWorldFdp(p, d2, q2);
        xt::xarray<double> F2_dx = SF2->getWorldFdx(p, d2, q2);
        xt::xarray<double> F2_dpdp = SF2->getWorldFdpdp(p, d2, q2);
        xt::xarray<double> F2_dpdx = SF2->getWorldFdpdx(p, d2, q2);
        xt::xarray<double> F2_dxdx = SF2->getWorldFdxdx(p, d2, q2);
        xt::xarray<double> F2_dpdpdp = SF2->getWorldFdpdpdp(p, d2, q2);
        xt::xarray<double> F2_dpdpdx = SF2->getWorldFdpdpdx(p, d2, q2);
        xt::xarray<double> F2_dpdxdx = SF2->getWorldFdpdxdx(p, d2, q2);

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx, alpha_dxdx;
        std::tie(alpha_dx, alpha_dxdx) = getGradientAndHessianGeneral(dual_var, 
            F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx, F1_dxdx, F2_dxdx,
            F1_dpdpdp, F2_dpdpdp, F1_dpdpdx, F2_dpdpdx, F1_dpdxdx, F2_dpdxdx);

        return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
    }
}

std::tuple<double, xt::xarray<double>> getGradient2d(
    const xt::xarray<double>& p, std::shared_ptr<ScalingFunction2d> SF1, const xt::xarray<double>& d1, double theta1,
    std::shared_ptr<ScalingFunction2d> SF2, const xt::xarray<double>& d2, double theta2){

    if (SF1->isMoving == false && SF2->isMoving == false){
        throw std::invalid_argument("Both scaling functions are not moving.");
    } else if (SF1->isMoving == true && SF2->isMoving == true){
        int dim_p = 2, dim_x1 = 3, dim_x2 = 3, dim_x = dim_x1 + dim_x2; // x = [x1,x2]
        double alpha = SF1->getWorldF(p, d1, theta1);


        xt::xarray<double> F1_dp, F1_dx1, F1_dpdp, F1_dpdx1;
        std::tie(F1_dp, F1_dx1, F1_dpdp, F1_dpdx1) = SF1->getWorldFFirstToSecondDers(p, d1, theta1);

        xt::xarray<double> F1_dx = xt::zeros<double>({dim_x});
        xt::view(F1_dx, xt::range(0, dim_x1)) = F1_dx1;
        xt::xarray<double> F1_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F1_dpdx, xt::all(), xt::range(0, dim_x1)) = F1_dpdx1;

        xt::xarray<double> F2_dp, F2_dx2, F2_dpdp, F2_dpdx2;
        std::tie(F2_dp, F2_dx2, F2_dpdp, F2_dpdx2) = SF2->getWorldFFirstToSecondDers(p, d2, theta2);

        xt::xarray<double> F2_dx = xt::zeros<double>({dim_x});
        xt::view(F2_dx, xt::range(dim_x1, dim_x)) = F2_dx2;
        xt::xarray<double> F2_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F2_dpdx, xt::all(), xt::range(dim_x1, dim_x)) = F2_dpdx2;
        xt::xarray<double> F2_dxdx = xt::zeros<double>({dim_x, dim_x});

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx = getGradientGeneral(dual_var, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx);

        return std::make_tuple(alpha, alpha_dx);
        
    } else { // one of the scaling functions is moving
        // int dim_p = 2, dim_x = 3;
        double alpha = SF1->getWorldF(p, d1, theta1);

        xt::xarray<double> F1_dp, F1_dx, F1_dpdp, F1_dpdx;
        std::tie(F1_dp, F1_dx, F1_dpdp, F1_dpdx) = SF1->getWorldFFirstToSecondDers(p, d1, theta1);

        xt::xarray<double> F2_dp, F2_dx, F2_dpdp, F2_dpdx;
        std::tie(F2_dp, F2_dx, F2_dpdp, F2_dpdx) = SF2->getWorldFFirstToSecondDers(p, d2, theta2);

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx = getGradientGeneral(dual_var, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx);

        return std::make_tuple(alpha, alpha_dx);
    }
}

std::tuple<double, xt::xarray<double>> getGradient3d(
    const xt::xarray<double>& p, std::shared_ptr<ScalingFunction3d> SF1, const xt::xarray<double>& d1, const xt::xarray<double>& q1,
    std::shared_ptr<ScalingFunction3d> SF2, const xt::xarray<double>& d2, const xt::xarray<double>& q2){

    if (SF1->isMoving == false && SF2->isMoving == false){
        throw std::invalid_argument("Both scaling functions are not moving.");
    } else if (SF1->isMoving == true && SF2->isMoving == true){
        int dim_p = 3, dim_x1 = 7, dim_x2 = 7, dim_x = dim_x1 + dim_x2; // x = [x1,x2]
        double alpha = SF1->getWorldF(p, d1, q1);

        xt::xarray<double> F1_dp, F1_dx1, F1_dpdp, F1_dpdx1;
        std::tie(F1_dp, F1_dx1, F1_dpdp, F1_dpdx1) = SF1->getWorldFFirstToSecondDers(p, d1, q1);

        xt::xarray<double> F1_dx = xt::zeros<double>({dim_x});
        xt::view(F1_dx, xt::range(0, dim_x1)) = F1_dx1;
        xt::xarray<double> F1_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F1_dpdx, xt::all(), xt::range(0, dim_x1)) = F1_dpdx1;

        xt::xarray<double> F2_dp, F2_dx2, F2_dpdp, F2_dpdx2;
        std::tie(F2_dp, F2_dx2, F2_dpdp, F2_dpdx2) = SF2->getWorldFFirstToSecondDers(p, d2, q2);

        xt::xarray<double> F2_dx = xt::zeros<double>({dim_x});
        xt::view(F2_dx, xt::range(dim_x1, dim_x)) = F2_dx2;
        xt::xarray<double> F2_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F2_dpdx, xt::all(), xt::range(dim_x1, dim_x)) = F2_dpdx2;
        xt::xarray<double> F2_dxdx = xt::zeros<double>({dim_x, dim_x});

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx = getGradientGeneral(dual_var, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx);

        return std::make_tuple(alpha, alpha_dx);
        
    } else { // one of the scaling functions is moving
        // int dim_p = 3, dim_x = 7;
        double alpha = SF1->getWorldF(p, d1, q1);

        xt::xarray<double> F1_dp, F1_dx, F1_dpdp, F1_dpdx;
        std::tie(F1_dp, F1_dx, F1_dpdp, F1_dpdx) = SF1->getWorldFFirstToSecondDers(p, d1, q1);

        xt::xarray<double> F2_dp, F2_dx, F2_dpdp, F2_dpdx;
        std::tie(F2_dp, F2_dx, F2_dpdp, F2_dpdx) = SF2->getWorldFFirstToSecondDers(p, d2, q2);

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx = getGradientGeneral(dual_var, F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx);

        return std::make_tuple(alpha, alpha_dx);
    }
}

std::tuple<double, xt::xarray<double>, xt::xarray<double>> getGradientAndHessian2d(
    const xt::xarray<double>& p, std::shared_ptr<ScalingFunction2d> SF1, const xt::xarray<double>& d1, double theta1,
    std::shared_ptr<ScalingFunction2d> SF2, const xt::xarray<double>& d2, double theta2){

    if (SF1->isMoving == false && SF2->isMoving == false){
        throw std::invalid_argument("Both scaling functions are not moving.");
    } else if (SF1->isMoving == true && SF2->isMoving == true){
        int dim_p = 2, dim_x1 = 3, dim_x2 = 3, dim_x = dim_x1 + dim_x2; // x = [x1,x2]
        double alpha = SF1->getWorldF(p, d1, theta1);

        xt::xarray<double> F1_dp, F1_dx1, F1_dpdp, F1_dpdx1, F1_dx1dx1, F1_dpdpdp, F1_dpdpdx1, F1_dpdx1dx1;
        std::tie(F1_dp, F1_dx1, F1_dpdp, F1_dpdx1, F1_dx1dx1, F1_dpdpdp, F1_dpdpdx1, F1_dpdx1dx1) 
            = SF1->getWorldFFirstToThirdDers(p, d1, theta1);

        xt::xarray<double> F1_dx = xt::zeros<double>({dim_x});
        xt::view(F1_dx, xt::range(0, dim_x1)) = F1_dx1;
        xt::xarray<double> F1_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F1_dpdx, xt::all(), xt::range(0, dim_x1)) = F1_dpdx1;
        xt::xarray<double> F1_dxdx = xt::zeros<double>({dim_x, dim_x});
        xt::view(F1_dxdx, xt::range(0, dim_x1), xt::range(0, dim_x1)) = F1_dx1dx1;
        xt::xarray<double> F1_dpdpdx = xt::zeros<double>({dim_p, dim_p, dim_x});
        xt::view(F1_dpdpdx, xt::all(), xt::all(), xt::range(0, dim_x1)) = F1_dpdpdx1;
        xt::xarray<double> F1_dpdxdx = xt::zeros<double>({dim_p, dim_x, dim_x});
        xt::view(F1_dpdxdx, xt::all(), xt::range(0, dim_x1), xt::range(0, dim_x1)) = F1_dpdx1dx1;

        xt::xarray<double> F2_dp, F2_dx2, F2_dpdp, F2_dpdx2, F2_dx2dx2, F2_dpdpdp, F2_dpdpdx2, F2_dpdx2dx2;
        std::tie(F2_dp, F2_dx2, F2_dpdp, F2_dpdx2, F2_dx2dx2, F2_dpdpdp, F2_dpdpdx2, F2_dpdx2dx2) 
            = SF2->getWorldFFirstToThirdDers(p, d2, theta2);

        xt::xarray<double> F2_dx = xt::zeros<double>({dim_x});
        xt::view(F2_dx, xt::range(dim_x1, dim_x)) = F2_dx2;
        xt::xarray<double> F2_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F2_dpdx, xt::all(), xt::range(dim_x1, dim_x)) = F2_dpdx2;
        xt::xarray<double> F2_dxdx = xt::zeros<double>({dim_x, dim_x});
        xt::view(F2_dxdx, xt::range(dim_x1, dim_x), xt::range(dim_x1, dim_x)) = F2_dx2dx2;
        xt::xarray<double> F2_dpdpdx = xt::zeros<double>({dim_p, dim_p, dim_x});
        xt::view(F2_dpdpdx, xt::all(), xt::all(), xt::range(dim_x1, dim_x)) = F2_dpdpdx2;
        xt::xarray<double> F2_dpdxdx = xt::zeros<double>({dim_p, dim_x, dim_x});
        xt::view(F2_dpdxdx, xt::all(), xt::range(dim_x1, dim_x), xt::range(dim_x1, dim_x)) = F2_dpdx2dx2;

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx, alpha_dxdx;
        std::tie(alpha_dx, alpha_dxdx) = getGradientAndHessianGeneral(dual_var, 
            F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx, F1_dxdx, F2_dxdx,
            F1_dpdpdp, F2_dpdpdp, F1_dpdpdx, F2_dpdpdx, F1_dpdxdx, F2_dpdxdx);
        
        return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
        
    } else { // one of the scaling functions is moving
        // int dim_p = 2, dim_x = 3;
        double alpha = SF1->getWorldF(p, d1, theta1);

        xt::xarray<double> F1_dp, F1_dx, F1_dpdp, F1_dpdx, F1_dxdx, F1_dpdpdp, F1_dpdpdx, F1_dpdxdx;
        std::tie(F1_dp, F1_dx, F1_dpdp, F1_dpdx, F1_dxdx, F1_dpdpdp, F1_dpdpdx, F1_dpdxdx) 
            = SF1->getWorldFFirstToThirdDers(p, d1, theta1);

        xt::xarray<double> F2_dp, F2_dx, F2_dpdp, F2_dpdx, F2_dxdx, F2_dpdpdp, F2_dpdpdx, F2_dpdxdx;
        std::tie(F2_dp, F2_dx, F2_dpdp, F2_dpdx, F2_dxdx, F2_dpdpdp, F2_dpdpdx, F2_dpdxdx) 
            = SF2->getWorldFFirstToThirdDers(p, d2, theta2);

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx, alpha_dxdx;
        std::tie(alpha_dx, alpha_dxdx) = getGradientAndHessianGeneral(dual_var, 
            F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx, F1_dxdx, F2_dxdx,
            F1_dpdpdp, F2_dpdpdp, F1_dpdpdx, F2_dpdpdx, F1_dpdxdx, F2_dpdxdx);

        return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
    }
}

std::tuple<double, xt::xarray<double>, xt::xarray<double>> getGradientAndHessian3d(
    const xt::xarray<double>& p, std::shared_ptr<ScalingFunction3d> SF1, const xt::xarray<double>& d1, const xt::xarray<double>& q1,
    std::shared_ptr<ScalingFunction3d> SF2, const xt::xarray<double>& d2, const xt::xarray<double>& q2){

    if (SF1->isMoving == false && SF2->isMoving == false){
        throw std::invalid_argument("Both scaling functions are not moving.");
    } else if (SF1->isMoving == true && SF2->isMoving == true){
        int dim_p = 3, dim_x1 = 7, dim_x2 = 7, dim_x = dim_x1 + dim_x2; // x = [x1,x2]
        double alpha = SF1->getWorldF(p, d1, q1);

        xt::xarray<double> F1_dp, F1_dx1, F1_dpdp, F1_dpdx1, F1_dx1dx1, F1_dpdpdp, F1_dpdpdx1, F1_dpdx1dx1;
        std::tie(F1_dp, F1_dx1, F1_dpdp, F1_dpdx1, F1_dx1dx1, F1_dpdpdp, F1_dpdpdx1, F1_dpdx1dx1) 
            = SF1->getWorldFFirstToThirdDers(p, d1, q1);

        xt::xarray<double> F1_dx = xt::zeros<double>({dim_x});
        xt::view(F1_dx, xt::range(0, dim_x1)) = F1_dx1;
        xt::xarray<double> F1_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F1_dpdx, xt::all(), xt::range(0, dim_x1)) = F1_dpdx1;
        xt::xarray<double> F1_dxdx = xt::zeros<double>({dim_x, dim_x});
        xt::view(F1_dxdx, xt::range(0, dim_x1), xt::range(0, dim_x1)) = F1_dx1dx1;
        xt::xarray<double> F1_dpdpdx = xt::zeros<double>({dim_p, dim_p, dim_x});
        xt::view(F1_dpdpdx, xt::all(), xt::all(), xt::range(0, dim_x1)) = F1_dpdpdx1;
        xt::xarray<double> F1_dpdxdx = xt::zeros<double>({dim_p, dim_x, dim_x});
        xt::view(F1_dpdxdx, xt::all(), xt::range(0, dim_x1), xt::range(0, dim_x1)) = F1_dpdx1dx1;

        xt::xarray<double> F2_dp, F2_dx2, F2_dpdp, F2_dpdx2, F2_dx2dx2, F2_dpdpdp, F2_dpdpdx2, F2_dpdx2dx2;
        std::tie(F2_dp, F2_dx2, F2_dpdp, F2_dpdx2, F2_dx2dx2, F2_dpdpdp, F2_dpdpdx2, F2_dpdx2dx2) 
            = SF2->getWorldFFirstToThirdDers(p, d2, q2);

        xt::xarray<double> F2_dx = xt::zeros<double>({dim_x});
        xt::view(F2_dx, xt::range(dim_x1, dim_x)) = F2_dx2;
        xt::xarray<double> F2_dpdx = xt::zeros<double>({dim_p, dim_x});
        xt::view(F2_dpdx, xt::all(), xt::range(dim_x1, dim_x)) = F2_dpdx2;
        xt::xarray<double> F2_dxdx = xt::zeros<double>({dim_x, dim_x});
        xt::view(F2_dxdx, xt::range(dim_x1, dim_x), xt::range(dim_x1, dim_x)) = F2_dx2dx2;
        xt::xarray<double> F2_dpdpdx = xt::zeros<double>({dim_p, dim_p, dim_x});
        xt::view(F2_dpdpdx, xt::all(), xt::all(), xt::range(dim_x1, dim_x)) = F2_dpdpdx2;
        xt::xarray<double> F2_dpdxdx = xt::zeros<double>({dim_p, dim_x, dim_x});
        xt::view(F2_dpdxdx, xt::all(), xt::range(dim_x1, dim_x), xt::range(dim_x1, dim_x)) = F2_dpdx2dx2;

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx, alpha_dxdx;
        std::tie(alpha_dx, alpha_dxdx) = getGradientAndHessianGeneral(dual_var, 
            F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx, F1_dxdx, F2_dxdx,
            F1_dpdpdp, F2_dpdpdp, F1_dpdpdx, F2_dpdpdx, F1_dpdxdx, F2_dpdxdx);
        
        return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
        
    } else { // one of the scaling functions is moving
        // int dim_p = 3, dim_x = 7;
        double alpha = SF1->getWorldF(p, d1, q1);

        xt::xarray<double> F1_dp, F1_dx, F1_dpdp, F1_dpdx, F1_dxdx, F1_dpdpdp, F1_dpdpdx, F1_dpdxdx;
        std::tie(F1_dp, F1_dx, F1_dpdp, F1_dpdx, F1_dxdx, F1_dpdpdp, F1_dpdpdx, F1_dpdxdx) 
            = SF1->getWorldFFirstToThirdDers(p, d1, q1);

        xt::xarray<double> F2_dp, F2_dx, F2_dpdp, F2_dpdx, F2_dxdx, F2_dpdpdp, F2_dpdpdx, F2_dpdxdx;
        std::tie(F2_dp, F2_dx, F2_dpdp, F2_dpdx, F2_dxdx, F2_dpdpdp, F2_dpdpdx, F2_dpdxdx) 
            = SF2->getWorldFFirstToThirdDers(p, d2, q2);

        double dual_var = getDualVariable(F1_dp, F2_dp);
        xt::xarray<double> alpha_dx, alpha_dxdx;
        std::tie(alpha_dx, alpha_dxdx) = getGradientAndHessianGeneral(dual_var, 
            F1_dp, F2_dp, F1_dx, F2_dx, F1_dpdp, F2_dpdp, F1_dpdx, F2_dpdx, F1_dxdx, F2_dxdx,
            F1_dpdpdp, F2_dpdpdp, F1_dpdpdx, F2_dpdpdx, F1_dpdxdx, F2_dpdxdx);

        return std::make_tuple(alpha, alpha_dx, alpha_dxdx);
    }
}
