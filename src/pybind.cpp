#include <numeric>
#include <xtensor.hpp>
#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include "rimonMethod.hpp"
#include "diffOptHelper.hpp"
#include "smoothMinimum.hpp"
#include "scalingFunction3d.hpp"
#include "ellipsoid3d.hpp"
#include "logSumExp3d.hpp"
#include "hyperplane3d.hpp"
#include "superquadrics3d.hpp"

namespace py = pybind11;

PYBIND11_MODULE(diffOptHelper2, m) {
    xt::import_numpy();
    m.doc() = "diffOptHelper2";

    m.def("rimonMethod", &rimonMethod, "rimonMethod based on xtensor.");

    m.def("getDualVariable", &getDualVariable, "getDualVariable based on xtensor.");
    m.def("getGradientGeneral", &getGradientGeneral, "getGradientGeneral based on xtensor.");
    m.def("getGradientAndHessianGeneral", &getGradientAndHessianGeneral, "getGradientAndHessianGeneral based on xtensor.");

    m.def("getSmoothMinimumAndLocalGradientAndHessian", &getSmoothMinimumAndLocalGradientAndHessian, "getSmoothMinimumLocalDerivatives based on xtensor");
    m.def("getSmoothMinimumAndTotalGradientAndHessian", &getSmoothMinimumAndTotalGradientAndHessian, "getSmoothMinimumGradientAndHessian based on xtensor");

    py::class_<ScalingFunction3d>(m, "ScalingFunction3d");

    py::class_<Ellipsoid3d, ScalingFunction3d>(m, "Ellipsoid3d")
        .def(py::init<bool, const xt::xarray<double>&, const xt::xarray<double>&>())
        .def("getBodyF", &Ellipsoid3d::getBodyF)
        .def("getBodyFdP", &Ellipsoid3d::getBodyFdP)
        .def("getBodyFdPdP", &Ellipsoid3d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &Ellipsoid3d::getBodyFdPdPdP)
        .def_readwrite("isMoving", &ScalingFunction3d::isMoving)
        .def("getBodyP", &ScalingFunction3d::getBodyP)
        .def("getBodyPdp", &ScalingFunction3d::getBodyPdp)
        .def("getBodyPdx", &ScalingFunction3d::getBodyPdx)
        .def("getBodyPdpdx", &ScalingFunction3d::getBodyPdpdx)
        .def("getBodyPdxdx", &ScalingFunction3d::getBodyPdxdx)
        .def("getBodyPdpdxdx", &ScalingFunction3d::getBodyPdpdxdx)
        .def("getWorldFdp", &ScalingFunction3d::getWorldFdp)
        .def("getWorldFdx", &ScalingFunction3d::getWorldFdx)
        .def("getWorldFdpdp", &ScalingFunction3d::getWorldFdpdp)
        .def("getWorldFdpdx", &ScalingFunction3d::getWorldFdpdx)
        .def("getWorldFdxdx", &ScalingFunction3d::getWorldFdxdx)
        .def("getWorldFdpdpdp", &ScalingFunction3d::getWorldFdpdpdp)
        .def("getWorldFdpdpdx", &ScalingFunction3d::getWorldFdpdpdx)
        .def("getWorldFdpdxdx", &ScalingFunction3d::getWorldFdpdxdx);


    py::class_<LogSumExp3d, ScalingFunction3d>(m, "LogSumExp3d")
        .def(py::init<bool, const xt::xarray<double>&, const xt::xarray<double>&, double>())
        .def("getBodyF", &LogSumExp3d::getBodyF)
        .def("getBodyFdP", &LogSumExp3d::getBodyFdP)
        .def("getBodyFdPdP", &LogSumExp3d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &LogSumExp3d::getBodyFdPdPdP)
        .def_readwrite("isMoving", &ScalingFunction3d::isMoving)
        .def("getBodyP", &ScalingFunction3d::getBodyP)
        .def("getBodyPdp", &ScalingFunction3d::getBodyPdp)
        .def("getBodyPdx", &ScalingFunction3d::getBodyPdx)
        .def("getBodyPdpdx", &ScalingFunction3d::getBodyPdpdx)
        .def("getBodyPdxdx", &ScalingFunction3d::getBodyPdxdx)
        .def("getBodyPdpdxdx", &ScalingFunction3d::getBodyPdpdxdx)
        .def("getWorldFdp", &ScalingFunction3d::getWorldFdp)
        .def("getWorldFdx", &ScalingFunction3d::getWorldFdx)
        .def("getWorldFdpdp", &ScalingFunction3d::getWorldFdpdp)
        .def("getWorldFdpdx", &ScalingFunction3d::getWorldFdpdx)
        .def("getWorldFdxdx", &ScalingFunction3d::getWorldFdxdx)
        .def("getWorldFdpdpdp", &ScalingFunction3d::getWorldFdpdpdp)
        .def("getWorldFdpdpdx", &ScalingFunction3d::getWorldFdpdpdx)
        .def("getWorldFdpdxdx", &ScalingFunction3d::getWorldFdpdxdx);
    
    py::class_<Hyperplane3d, ScalingFunction3d>(m, "Hyperplane3d")
        .def(py::init<bool, const xt::xarray<double>&, double>())
        .def("getBodyF", &Hyperplane3d::getBodyF)
        .def("getBodyFdP", &Hyperplane3d::getBodyFdP)
        .def("getBodyFdPdP", &Hyperplane3d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &Hyperplane3d::getBodyFdPdPdP)
        .def_readwrite("isMoving", &ScalingFunction3d::isMoving)
        .def("getBodyP", &ScalingFunction3d::getBodyP)
        .def("getBodyPdp", &ScalingFunction3d::getBodyPdp)
        .def("getBodyPdx", &ScalingFunction3d::getBodyPdx)
        .def("getBodyPdpdx", &ScalingFunction3d::getBodyPdpdx)
        .def("getBodyPdxdx", &ScalingFunction3d::getBodyPdxdx)
        .def("getBodyPdpdxdx", &ScalingFunction3d::getBodyPdpdxdx)
        .def("getWorldFdp", &ScalingFunction3d::getWorldFdp)
        .def("getWorldFdx", &ScalingFunction3d::getWorldFdx)
        .def("getWorldFdpdp", &ScalingFunction3d::getWorldFdpdp)
        .def("getWorldFdpdx", &ScalingFunction3d::getWorldFdpdx)
        .def("getWorldFdxdx", &ScalingFunction3d::getWorldFdxdx)
        .def("getWorldFdpdpdp", &ScalingFunction3d::getWorldFdpdpdp)
        .def("getWorldFdpdpdx", &ScalingFunction3d::getWorldFdpdpdx)
        .def("getWorldFdpdxdx", &ScalingFunction3d::getWorldFdpdxdx);

    py::class_<Superquadrics3d, ScalingFunction3d>(m, "Superquadrics3d")
        .def(py::init<bool, const xt::xarray<double>&, const xt::xarray<double>&, double, double>())
        .def("getBodyF", &Superquadrics3d::getBodyF)
        .def("getBodyFdP", &Superquadrics3d::getBodyFdP)
        .def("getBodyFdPdP", &Superquadrics3d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &Superquadrics3d::getBodyFdPdPdP)
        .def_readwrite("isMoving", &ScalingFunction3d::isMoving)
        .def("getBodyP", &ScalingFunction3d::getBodyP)
        .def("getBodyPdp", &ScalingFunction3d::getBodyPdp)
        .def("getBodyPdx", &ScalingFunction3d::getBodyPdx)
        .def("getBodyPdpdx", &ScalingFunction3d::getBodyPdpdx)
        .def("getBodyPdxdx", &ScalingFunction3d::getBodyPdxdx)
        .def("getBodyPdpdxdx", &ScalingFunction3d::getBodyPdpdxdx)
        .def("getWorldFdp", &ScalingFunction3d::getWorldFdp)
        .def("getWorldFdx", &ScalingFunction3d::getWorldFdx)
        .def("getWorldFdpdp", &ScalingFunction3d::getWorldFdpdp)
        .def("getWorldFdpdx", &ScalingFunction3d::getWorldFdpdx)
        .def("getWorldFdxdx", &ScalingFunction3d::getWorldFdxdx)
        .def("getWorldFdpdpdp", &ScalingFunction3d::getWorldFdpdpdp)
        .def("getWorldFdpdpdx", &ScalingFunction3d::getWorldFdpdpdx)
        .def("getWorldFdpdxdx", &ScalingFunction3d::getWorldFdpdxdx);
}
