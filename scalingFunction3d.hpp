#include <xtensor/xarray.hpp>

class ScalingFunction3d {
    public:
        ScalingFunction3d() = default;
        ~ScalingFunction3d() = default;
        // virtual double getBodyF(const xt::xarray<double>& P) const = 0;
        // virtual xt::xarray<double> getBodyFdP(const xt::xarray<double>& P) const = 0;
        // virtual xt::xarray<double> getBodyFdPdP(const xt::xarray<double>& P) const = 0;
        // virtual xt::xarray<double> getBodyFdPdPdP(const xt::xarray<double>& P) const = 0;
        xt::xarray<double> getBodyPdp(const xt::xarray<double>& R);
        xt::xarray<double> getBodyPdx(const xt::xarray<double>& p, const xt::xarray<double>& d,
            const xt::xarray<double>& q, const xt::xarray<double>& R);
        xt::xarray<double> getBodyPdpdx(const xt::xarray<double>& q);
        xt::xarray<double> getBodyPdxdx(const xt::xarray<double>& p, const xt::xarray<double>& d, const xt::xarray<double>& q);
        xt::xarray<double> getBodyPdpdxdx();

};