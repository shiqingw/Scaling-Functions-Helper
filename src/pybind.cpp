#include <numeric>
#include <xtensor.hpp>
#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include "diffOptHelper.hpp"
#include "smoothMinimum.hpp"
#include "scalingFunction3d.hpp"

namespace py = pybind11;

PYBIND11_MODULE(diffOptHelper2, m) {
    xt::import_numpy();
    m.doc() = "diffOptHelper2";

    m.def("getDualVariable", &getDualVariable, "getDualVariable based on xtensor.");
    m.def("getGradientGeneral", &getGradientGeneral, "getGradientGeneral based on xtensor.");
    m.def("getGradientAndHessianGeneral", &getGradientAndHessianGeneral, "getGradientAndHessianGeneral based on xtensor.");

    m.def("getSmoothMinimumAndLocalGradientAndHessian", &getSmoothMinimumAndLocalGradientAndHessian, "getSmoothMinimumLocalDerivatives based on xtensor");
    m.def("getSmoothMinimumAndTotalGradientAndHessian", &getSmoothMinimumAndTotalGradientAndHessian, "getSmoothMinimumGradientAndHessian based on xtensor");

    py::class_<ScalingFunction3d>(m, "ScalingFunction3d")
        .def(py::init<>())
        .def("getBodyPdp", &ScalingFunction3d::getBodyPdp)
        .def("getBodyPdx", &ScalingFunction3d::getBodyPdx)
        .def("getBodyPdpdx", &ScalingFunction3d::getBodyPdpdx)
        .def("getBodyPdxdx", &ScalingFunction3d::getBodyPdxdx)
        .def("getBodyPdpdxdx", &ScalingFunction3d::getBodyPdpdxdx);

}
