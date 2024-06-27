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
#include "scalingFunction2d.hpp"
#include "ellipsoid2d.hpp"
#include "logSumExp2d.hpp"
#include "hyperplane2d.hpp"

namespace py = pybind11;

PYBIND11_MODULE(scalingFunctionsHelper, m) {
    xt::import_numpy();
    m.doc() = "scalingFunctionsHelper";

    m.def("rimonMethod", &rimonMethod, "rimonMethod based on xtensor.");
    m.def("rimonMethod2d", &rimonMethod2d, "rimonMethod2d based on xtensor.");
    m.def("rimonMethod3d", &rimonMethod3d, "rimonMethod3d based on xtensor.");

    m.def("getDualVariable", &getDualVariable, "getDualVariable based on xtensor.");
    m.def("getGradientGeneral", &getGradientGeneral, "getGradientGeneral based on xtensor.");
    m.def("getGradientAndHessianGeneral", &getGradientAndHessianGeneral, "getGradientAndHessianGeneral based on xtensor.");
    m.def("getGradient2d", &getGradient2d, "getGradient2d based on xtensor.");
    m.def("getGradient3d", &getGradient3d, "getGradient3d based on xtensor.");
    m.def("getGradientAndHessian2d", &getGradientAndHessian2d, "getGradientAndHessian2d based on xtensor.");
    m.def("getGradientAndHessian3d", &getGradientAndHessian3d, "getGradientAndHessian3d based on xtensor.");

    m.def("getSmoothMinimumAndLocalGradientAndHessian", &getSmoothMinimumAndLocalGradientAndHessian, "getSmoothMinimumLocalDerivatives based on xtensor");
    m.def("getSmoothMinimumAndTotalGradientAndHessian", &getSmoothMinimumAndTotalGradientAndHessian, "getSmoothMinimumGradientAndHessian based on xtensor");

    py::class_<ScalingFunction3d>(m, "ScalingFunction3d");

    py::class_<Ellipsoid3d, ScalingFunction3d>(m, "Ellipsoid3d")
        .def(py::init<bool, const xt::xarray<double>&, const xt::xarray<double>&>())
        .def("getBodyF", &Ellipsoid3d::getBodyF)
        .def("getBodyFdP", &Ellipsoid3d::getBodyFdP)
        .def("getBodyFdPdP", &Ellipsoid3d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &Ellipsoid3d::getBodyFdPdPdP)
        .def("getWorldQuadraticCoefficient", &Ellipsoid3d::getWorldQuadraticCoefficient)
        .def("getWorldCenter", &Ellipsoid3d::getWorldCenter)
        .def_readwrite("isMoving", &ScalingFunction3d::isMoving)
        .def("getRotationMatrix", &ScalingFunction3d::getRotationMatrix)
        .def("getBodyP", &ScalingFunction3d::getBodyP)
        .def("getBodyPdp", &ScalingFunction3d::getBodyPdp)
        .def("getBodyPdx", &ScalingFunction3d::getBodyPdx)
        .def("getBodyPdpdx", &ScalingFunction3d::getBodyPdpdx)
        .def("getBodyPdxdx", &ScalingFunction3d::getBodyPdxdx)
        .def("getBodyPdpdxdx", &ScalingFunction3d::getBodyPdpdxdx)
        .def("getWorldF", &ScalingFunction3d::getWorldF)
        .def("getWorldFdp", &ScalingFunction3d::getWorldFdp)
        .def("getWorldFdx", &ScalingFunction3d::getWorldFdx)
        .def("getWorldFdpdp", &ScalingFunction3d::getWorldFdpdp)
        .def("getWorldFdpdx", &ScalingFunction3d::getWorldFdpdx)
        .def("getWorldFdxdx", &ScalingFunction3d::getWorldFdxdx)
        .def("getWorldFdpdpdp", &ScalingFunction3d::getWorldFdpdpdp)
        .def("getWorldFdpdpdx", &ScalingFunction3d::getWorldFdpdpdx)
        .def("getWorldFdpdxdx", &ScalingFunction3d::getWorldFdpdxdx)
        .def(py::pickle(
            [](const Ellipsoid3d &e) { // __getstate__
                return py::make_tuple(e.isMoving, e.Q, e.mu);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3) {
                    throw std::runtime_error("Invalid state!");
                }
                return Ellipsoid3d(t[0].cast<bool>(), t[1].cast<xt::xarray<double>>(), t[2].cast<xt::xarray<double>>());
            }
        ));

    py::class_<LogSumExp3d, ScalingFunction3d>(m, "LogSumExp3d")
        .def(py::init<bool, const xt::xarray<double>&, const xt::xarray<double>&, double>())
        .def("getBodyF", &LogSumExp3d::getBodyF)
        .def("getBodyFdP", &LogSumExp3d::getBodyFdP)
        .def("getBodyFdPdP", &LogSumExp3d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &LogSumExp3d::getBodyFdPdPdP)
        .def_readwrite("isMoving", &ScalingFunction3d::isMoving)
        .def("getRotationMatrix", &ScalingFunction3d::getRotationMatrix)
        .def("getBodyP", &ScalingFunction3d::getBodyP)
        .def("getBodyPdp", &ScalingFunction3d::getBodyPdp)
        .def("getBodyPdx", &ScalingFunction3d::getBodyPdx)
        .def("getBodyPdpdx", &ScalingFunction3d::getBodyPdpdx)
        .def("getBodyPdxdx", &ScalingFunction3d::getBodyPdxdx)
        .def("getBodyPdpdxdx", &ScalingFunction3d::getBodyPdpdxdx)
        .def("getWorldF", &ScalingFunction3d::getWorldF)
        .def("getWorldFdp", &ScalingFunction3d::getWorldFdp)
        .def("getWorldFdx", &ScalingFunction3d::getWorldFdx)
        .def("getWorldFdpdp", &ScalingFunction3d::getWorldFdpdp)
        .def("getWorldFdpdx", &ScalingFunction3d::getWorldFdpdx)
        .def("getWorldFdxdx", &ScalingFunction3d::getWorldFdxdx)
        .def("getWorldFdpdpdp", &ScalingFunction3d::getWorldFdpdpdp)
        .def("getWorldFdpdpdx", &ScalingFunction3d::getWorldFdpdpdx)
        .def("getWorldFdpdxdx", &ScalingFunction3d::getWorldFdpdxdx)
        .def(py::pickle(
            [](const LogSumExp3d &l) { // __getstate__
                return py::make_tuple(l.isMoving, l.A, l.b, l.kappa);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4) {
                    throw std::runtime_error("Invalid state!");
                }
                return LogSumExp3d(t[0].cast<bool>(), t[1].cast<xt::xarray<double>>(), t[2].cast<xt::xarray<double>>(), t[3].cast<double>());
            }
        ));
    
    py::class_<Hyperplane3d, ScalingFunction3d>(m, "Hyperplane3d")
        .def(py::init<bool, const xt::xarray<double>&, double>())
        .def("getBodyF", &Hyperplane3d::getBodyF)
        .def("getBodyFdP", &Hyperplane3d::getBodyFdP)
        .def("getBodyFdPdP", &Hyperplane3d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &Hyperplane3d::getBodyFdPdPdP)
        .def_readwrite("isMoving", &ScalingFunction3d::isMoving)
        .def("getRotationMatrix", &ScalingFunction3d::getRotationMatrix)
        .def("getBodyP", &ScalingFunction3d::getBodyP)
        .def("getBodyPdp", &ScalingFunction3d::getBodyPdp)
        .def("getBodyPdx", &ScalingFunction3d::getBodyPdx)
        .def("getBodyPdpdx", &ScalingFunction3d::getBodyPdpdx)
        .def("getBodyPdxdx", &ScalingFunction3d::getBodyPdxdx)
        .def("getBodyPdpdxdx", &ScalingFunction3d::getBodyPdpdxdx)
        .def("getWorldF", &ScalingFunction3d::getWorldF)
        .def("getWorldFdp", &ScalingFunction3d::getWorldFdp)
        .def("getWorldFdx", &ScalingFunction3d::getWorldFdx)
        .def("getWorldFdpdp", &ScalingFunction3d::getWorldFdpdp)
        .def("getWorldFdpdx", &ScalingFunction3d::getWorldFdpdx)
        .def("getWorldFdxdx", &ScalingFunction3d::getWorldFdxdx)
        .def("getWorldFdpdpdp", &ScalingFunction3d::getWorldFdpdpdp)
        .def("getWorldFdpdpdx", &ScalingFunction3d::getWorldFdpdpdx)
        .def("getWorldFdpdxdx", &ScalingFunction3d::getWorldFdpdxdx)
        .def(py::pickle(
            [](const Hyperplane3d &h) { // __getstate__
                return py::make_tuple(h.isMoving, h.a, h.b);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3) {
                    throw std::runtime_error("Invalid state!");
                }
                return Hyperplane3d(t[0].cast<bool>(), t[1].cast<xt::xarray<double>>(), t[2].cast<double>());
            }
        ));

    py::class_<Superquadrics3d, ScalingFunction3d>(m, "Superquadrics3d")
        .def(py::init<bool, const xt::xarray<double>&, const xt::xarray<double>&, double, double>())
        .def("getBodyF", &Superquadrics3d::getBodyF)
        .def("getBodyFdP", &Superquadrics3d::getBodyFdP)
        .def("getBodyFdPdP", &Superquadrics3d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &Superquadrics3d::getBodyFdPdPdP)
        .def_readwrite("isMoving", &ScalingFunction3d::isMoving)
        .def("getRotationMatrix", &ScalingFunction3d::getRotationMatrix)
        .def("getBodyP", &ScalingFunction3d::getBodyP)
        .def("getBodyPdp", &ScalingFunction3d::getBodyPdp)
        .def("getBodyPdx", &ScalingFunction3d::getBodyPdx)
        .def("getBodyPdpdx", &ScalingFunction3d::getBodyPdpdx)
        .def("getBodyPdxdx", &ScalingFunction3d::getBodyPdxdx)
        .def("getBodyPdpdxdx", &ScalingFunction3d::getBodyPdpdxdx)
        .def("getWorldF", &ScalingFunction3d::getWorldF)
        .def("getWorldFdp", &ScalingFunction3d::getWorldFdp)
        .def("getWorldFdx", &ScalingFunction3d::getWorldFdx)
        .def("getWorldFdpdp", &ScalingFunction3d::getWorldFdpdp)
        .def("getWorldFdpdx", &ScalingFunction3d::getWorldFdpdx)
        .def("getWorldFdxdx", &ScalingFunction3d::getWorldFdxdx)
        .def("getWorldFdpdpdp", &ScalingFunction3d::getWorldFdpdpdp)
        .def("getWorldFdpdpdx", &ScalingFunction3d::getWorldFdpdpdx)
        .def("getWorldFdpdxdx", &ScalingFunction3d::getWorldFdpdxdx)
        .def(py::pickle(
            [](const Superquadrics3d &s) { // __getstate__
                return py::make_tuple(s.isMoving, s.c, s.a, s.e1, s.e2);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 5) {
                    throw std::runtime_error("Invalid state!");
                }
                return Superquadrics3d(t[0].cast<bool>(), t[1].cast<xt::xarray<double>>(), t[2].cast<xt::xarray<double>>(), 
                        t[3].cast<double>(), t[4].cast<double>());
            }
        ));

    py::class_<ScalingFunction2d>(m, "ScalingFunction2d");

    py::class_<Ellipsoid2d, ScalingFunction2d>(m, "Ellipsoid2d")
        .def(py::init<bool, const xt::xarray<double>&, const xt::xarray<double>&>())
        .def("getBodyF", &Ellipsoid2d::getBodyF)
        .def("getBodyFdP", &Ellipsoid2d::getBodyFdP)
        .def("getBodyFdPdP", &Ellipsoid2d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &Ellipsoid2d::getBodyFdPdPdP)
        .def("getWorldQuadraticCoefficient", &Ellipsoid2d::getWorldQuadraticCoefficient)
        .def("getWorldCenter", &Ellipsoid2d::getWorldCenter)
        .def_readwrite("isMoving", &ScalingFunction2d::isMoving)
        .def("getRotationMatrix", &ScalingFunction2d::getRotationMatrix)
        .def("getBodyP", &ScalingFunction2d::getBodyP)
        .def("getBodyPdp", &ScalingFunction2d::getBodyPdp)
        .def("getBodyPdx", &ScalingFunction2d::getBodyPdx)
        .def("getBodyPdpdx", &ScalingFunction2d::getBodyPdpdx)
        .def("getBodyPdxdx", &ScalingFunction2d::getBodyPdxdx)
        .def("getBodyPdpdxdx", &ScalingFunction2d::getBodyPdpdxdx)
        .def("getWorldF", &ScalingFunction2d::getWorldF)
        .def("getWorldFdp", &ScalingFunction2d::getWorldFdp)
        .def("getWorldFdx", &ScalingFunction2d::getWorldFdx)
        .def("getWorldFdpdp", &ScalingFunction2d::getWorldFdpdp)
        .def("getWorldFdpdx", &ScalingFunction2d::getWorldFdpdx)
        .def("getWorldFdxdx", &ScalingFunction2d::getWorldFdxdx)
        .def("getWorldFdpdpdp", &ScalingFunction2d::getWorldFdpdpdp)
        .def("getWorldFdpdpdx", &ScalingFunction2d::getWorldFdpdpdx)
        .def("getWorldFdpdxdx", &ScalingFunction2d::getWorldFdpdxdx)
        .def(py::pickle(
            [](const Ellipsoid2d &e) { // __getstate__
                return py::make_tuple(e.isMoving, e.Q, e.mu);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3) {
                    throw std::runtime_error("Invalid state!");
                }
                return Ellipsoid2d(t[0].cast<bool>(), t[1].cast<xt::xarray<double>>(), t[2].cast<xt::xarray<double>>());
            }
        ));


    py::class_<LogSumExp2d, ScalingFunction2d>(m, "LogSumExp2d")
        .def(py::init<bool, const xt::xarray<double>&, const xt::xarray<double>&, double>())
        .def("getBodyF", &LogSumExp2d::getBodyF)
        .def("getBodyFdP", &LogSumExp2d::getBodyFdP)
        .def("getBodyFdPdP", &LogSumExp2d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &LogSumExp2d::getBodyFdPdPdP)
        .def_readwrite("isMoving", &ScalingFunction2d::isMoving)
        .def("getRotationMatrix", &ScalingFunction2d::getRotationMatrix)
        .def("getBodyP", &ScalingFunction2d::getBodyP)
        .def("getBodyPdp", &ScalingFunction2d::getBodyPdp)
        .def("getBodyPdx", &ScalingFunction2d::getBodyPdx)
        .def("getBodyPdpdx", &ScalingFunction2d::getBodyPdpdx)
        .def("getBodyPdxdx", &ScalingFunction2d::getBodyPdxdx)
        .def("getBodyPdpdxdx", &ScalingFunction2d::getBodyPdpdxdx)
        .def("getWorldF", &ScalingFunction2d::getWorldF)
        .def("getWorldFdp", &ScalingFunction2d::getWorldFdp)
        .def("getWorldFdx", &ScalingFunction2d::getWorldFdx)
        .def("getWorldFdpdp", &ScalingFunction2d::getWorldFdpdp)
        .def("getWorldFdpdx", &ScalingFunction2d::getWorldFdpdx)
        .def("getWorldFdxdx", &ScalingFunction2d::getWorldFdxdx)
        .def("getWorldFdpdpdp", &ScalingFunction2d::getWorldFdpdpdp)
        .def("getWorldFdpdpdx", &ScalingFunction2d::getWorldFdpdpdx)
        .def("getWorldFdpdxdx", &ScalingFunction2d::getWorldFdpdxdx)
        .def(py::pickle(
            [](const LogSumExp2d &l) { // __getstate__
                return py::make_tuple(l.isMoving, l.A, l.b, l.kappa);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4) {
                    throw std::runtime_error("Invalid state!");
                }
                return LogSumExp2d(t[0].cast<bool>(), t[1].cast<xt::xarray<double>>(), t[2].cast<xt::xarray<double>>(), t[3].cast<double>());
            }
        ));
    
    py::class_<Hyperplane2d, ScalingFunction2d>(m, "Hyperplane2d")
        .def(py::init<bool, const xt::xarray<double>&, double>())
        .def("getBodyF", &Hyperplane2d::getBodyF)
        .def("getBodyFdP", &Hyperplane2d::getBodyFdP)
        .def("getBodyFdPdP", &Hyperplane2d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &Hyperplane2d::getBodyFdPdPdP)
        .def_readwrite("isMoving", &ScalingFunction2d::isMoving)
        .def("getRotationMatrix", &ScalingFunction2d::getRotationMatrix)
        .def("getBodyP", &ScalingFunction2d::getBodyP)
        .def("getBodyPdp", &ScalingFunction2d::getBodyPdp)
        .def("getBodyPdx", &ScalingFunction2d::getBodyPdx)
        .def("getBodyPdpdx", &ScalingFunction2d::getBodyPdpdx)
        .def("getBodyPdxdx", &ScalingFunction2d::getBodyPdxdx)
        .def("getBodyPdpdxdx", &ScalingFunction2d::getBodyPdpdxdx)
        .def("getWorldF", &ScalingFunction2d::getWorldF)
        .def("getWorldFdp", &ScalingFunction2d::getWorldFdp)
        .def("getWorldFdx", &ScalingFunction2d::getWorldFdx)
        .def("getWorldFdpdp", &ScalingFunction2d::getWorldFdpdp)
        .def("getWorldFdpdx", &ScalingFunction2d::getWorldFdpdx)
        .def("getWorldFdxdx", &ScalingFunction2d::getWorldFdxdx)
        .def("getWorldFdpdpdp", &ScalingFunction2d::getWorldFdpdpdp)
        .def("getWorldFdpdpdx", &ScalingFunction2d::getWorldFdpdpdx)
        .def("getWorldFdpdxdx", &ScalingFunction2d::getWorldFdpdxdx)
        .def(py::pickle(
            [](const Hyperplane2d &h) { // __getstate__
                return py::make_tuple(h.isMoving, h.a, h.b);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3) {
                    throw std::runtime_error("Invalid state!");
                }
                return Hyperplane2d(t[0].cast<bool>(), t[1].cast<xt::xarray<double>>(), t[2].cast<double>());
            }
        ));
}
