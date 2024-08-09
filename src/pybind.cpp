#include <numeric>
#include <memory> 
#include <xtensor.hpp>
#include <pybind11/stl.h>
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

PYBIND11_MODULE(scalingFunctionsHelperPy, m) {
    xt::import_numpy();
    m.doc() = "scalingFunctionsHelperPy";

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

    py::class_<ScalingFunction3d, std::shared_ptr<ScalingFunction3d>>(m, "ScalingFunction3d");

    py::class_<Ellipsoid3d, ScalingFunction3d, std::shared_ptr<Ellipsoid3d>>(m, "Ellipsoid3d")
        .def(py::init<bool, const xt::xtensor<double, 2>&, const xt::xtensor<double, 1>&>())
        .def_readwrite("Q", &Ellipsoid3d::Q)
        .def_readwrite("mu", &Ellipsoid3d::mu)
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
        .def("getWorldFFirstToSecondDers", &ScalingFunction3d::getWorldFFirstToSecondDers)
        .def("getWorldFFirstToThirdDers", &ScalingFunction3d::getWorldFFirstToThirdDers)
        .def(py::pickle(
            [](const Ellipsoid3d &e) { // __getstate__
                return py::make_tuple(e.isMoving, e.Q, e.mu);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3) {
                    throw std::runtime_error("Invalid state!");
                }
                return std::make_shared<Ellipsoid3d>(t[0].cast<bool>(), t[1].cast<xt::xtensor<double, 2>>(), t[2].cast<xt::xtensor<double, 1>>());
            }
        ));

    py::class_<LogSumExp3d, ScalingFunction3d, std::shared_ptr<LogSumExp3d>>(m, "LogSumExp3d")
        .def(py::init<bool, const xt::xtensor<double,2>&, const xt::xtensor<double,1>&, double>())
        .def_readwrite("A", &LogSumExp3d::A)
        .def_readwrite("b", &LogSumExp3d::b)
        .def_readwrite("kappa", &LogSumExp3d::kappa)
        .def("getBodyF", &LogSumExp3d::getBodyF)
        .def("getBodyFdP", &LogSumExp3d::getBodyFdP)
        .def("getBodyFdPdP", &LogSumExp3d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &LogSumExp3d::getBodyFdPdPdP)
        .def("getWorldMatrixCoefficient", &LogSumExp3d::getWorldMatrixCoefficient)
        .def("getWorldVectorCoefficient", &LogSumExp3d::getWorldVectorCoefficient)
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
        .def("getWorldFFirstToSecondDers", &ScalingFunction3d::getWorldFFirstToSecondDers)
        .def("getWorldFFirstToThirdDers", &ScalingFunction3d::getWorldFFirstToThirdDers)
        .def(py::pickle(
            [](const LogSumExp3d &l) { // __getstate__
                return py::make_tuple(l.isMoving, l.A, l.b, l.kappa);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4) {
                    throw std::runtime_error("Invalid state!");
                }
                return std::make_shared<LogSumExp3d>(t[0].cast<bool>(), t[1].cast<xt::xtensor<double,2>>(), t[2].cast<xt::xtensor<double,1>>(), t[3].cast<double>());
            }
        ));
    
    py::class_<Hyperplane3d, ScalingFunction3d, std::shared_ptr<Hyperplane3d>>(m, "Hyperplane3d")
        .def(py::init<bool, const xt::xtensor<double, 1>&, double>())
        .def_readwrite("a", &Hyperplane3d::a)
        .def_readwrite("b", &Hyperplane3d::b)
        .def("getBodyF", &Hyperplane3d::getBodyF)
        .def("getBodyFdP", &Hyperplane3d::getBodyFdP)
        .def("getBodyFdPdP", &Hyperplane3d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &Hyperplane3d::getBodyFdPdPdP)
        .def("getWorldSlope", &Hyperplane3d::getWorldSlope)
        .def("getWorldOffset", &Hyperplane3d::getWorldOffset)
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
        .def("getWorldFFirstToSecondDers", &ScalingFunction3d::getWorldFFirstToSecondDers)
        .def("getWorldFFirstToThirdDers", &ScalingFunction3d::getWorldFFirstToThirdDers)
        .def(py::pickle(
            [](const Hyperplane3d &h) { // __getstate__
                return py::make_tuple(h.isMoving, h.a, h.b);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3) {
                    throw std::runtime_error("Invalid state!");
                }
                return std::make_shared<Hyperplane3d>(t[0].cast<bool>(), t[1].cast<xt::xtensor<double, 1>>(), t[2].cast<double>());
            }
        ));

    py::class_<Superquadrics3d, ScalingFunction3d, std::shared_ptr<Superquadrics3d>>(m, "Superquadrics3d")
        .def(py::init<bool, const xt::xtensor<double, 1>&, const xt::xtensor<double, 1>&, double, double>())
        .def_readwrite("c", &Superquadrics3d::c)
        .def_readwrite("a", &Superquadrics3d::a)
        .def_readwrite("e1", &Superquadrics3d::e1)
        .def_readwrite("e2", &Superquadrics3d::e2)
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
        .def("getWorldFFirstToSecondDers", &ScalingFunction3d::getWorldFFirstToSecondDers)
        .def("getWorldFFirstToThirdDers", &ScalingFunction3d::getWorldFFirstToThirdDers)
        .def(py::pickle(
            [](const Superquadrics3d &s) { // __getstate__
                return py::make_tuple(s.isMoving, s.c, s.a, s.e1, s.e2);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 5) {
                    throw std::runtime_error("Invalid state!");
                }
                return std::make_shared<Superquadrics3d>(t[0].cast<bool>(), t[1].cast<xt::xtensor<double, 1>>(), t[2].cast<xt::xtensor<double, 1>>(), 
                        t[3].cast<double>(), t[4].cast<double>());
            }
        ));

    py::class_<ScalingFunction2d, std::shared_ptr<ScalingFunction2d>>(m, "ScalingFunction2d");

    py::class_<Ellipsoid2d, ScalingFunction2d, std::shared_ptr<Ellipsoid2d>>(m, "Ellipsoid2d")
        .def(py::init<bool, const xt::xtensor<double, 2>&, const xt::xtensor<double, 1>&>())
        .def_readwrite("Q", &Ellipsoid2d::Q)
        .def_readwrite("mu", &Ellipsoid2d::mu)
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
        .def("getWorldFFirstToSecondDers", &ScalingFunction2d::getWorldFFirstToSecondDers)
        .def("getWorldFFirstToThirdDers", &ScalingFunction2d::getWorldFFirstToThirdDers)
        .def(py::pickle(
            [](const Ellipsoid2d &e) { // __getstate__
                return py::make_tuple(e.isMoving, e.Q, e.mu);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3) {
                    throw std::runtime_error("Invalid state!");
                }
                return std::make_shared<Ellipsoid2d>(t[0].cast<bool>(), t[1].cast<xt::xtensor<double, 2>>(), t[2].cast<xt::xtensor<double, 1>>());
            }
        ));


    py::class_<LogSumExp2d, ScalingFunction2d, std::shared_ptr<LogSumExp2d>>(m, "LogSumExp2d")
        .def(py::init<bool, const xt::xtensor<double, 2>&, const xt::xtensor<double, 1>&, double>())
        .def_readwrite("A", &LogSumExp2d::A)
        .def_readwrite("b", &LogSumExp2d::b)
        .def_readwrite("kappa", &LogSumExp2d::kappa)
        .def("getBodyF", &LogSumExp2d::getBodyF)
        .def("getBodyFdP", &LogSumExp2d::getBodyFdP)
        .def("getBodyFdPdP", &LogSumExp2d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &LogSumExp2d::getBodyFdPdPdP)
        .def("getWorldMatrixCoefficient", &LogSumExp2d::getWorldMatrixCoefficient)
        .def("getWorldVectorCoefficient", &LogSumExp2d::getWorldVectorCoefficient)
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
        .def("getWorldFFirstToSecondDers", &ScalingFunction2d::getWorldFFirstToSecondDers)
        .def("getWorldFFirstToThirdDers", &ScalingFunction2d::getWorldFFirstToThirdDers)
        .def(py::pickle(
            [](const LogSumExp2d &l) { // __getstate__
                return py::make_tuple(l.isMoving, l.A, l.b, l.kappa);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 4) {
                    throw std::runtime_error("Invalid state!");
                }
                return std::make_shared<LogSumExp2d>(t[0].cast<bool>(), t[1].cast<xt::xtensor<double, 2>>(), t[2].cast<xt::xtensor<double, 1>>(), t[3].cast<double>());
            }
        ));
    
    py::class_<Hyperplane2d, ScalingFunction2d, std::shared_ptr<Hyperplane2d>>(m, "Hyperplane2d")
        .def(py::init<bool, const xt::xtensor<double, 1>&, double>())
        .def_readwrite("a", &Hyperplane2d::a)
        .def_readwrite("b", &Hyperplane2d::b)
        .def("getBodyF", &Hyperplane2d::getBodyF)
        .def("getBodyFdP", &Hyperplane2d::getBodyFdP)
        .def("getBodyFdPdP", &Hyperplane2d::getBodyFdPdP)
        .def("getBodyFdPdPdP", &Hyperplane2d::getBodyFdPdPdP)
        .def("getWorldSlope", &Hyperplane2d::getWorldSlope)
        .def("getWorldOffset", &Hyperplane2d::getWorldOffset)
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
        .def("getWorldFFirstToSecondDers", &ScalingFunction2d::getWorldFFirstToSecondDers)
        .def("getWorldFFirstToThirdDers", &ScalingFunction2d::getWorldFFirstToThirdDers)
        .def(py::pickle(
            [](const Hyperplane2d &h) { // __getstate__
                return py::make_tuple(h.isMoving, h.a, h.b);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3) {
                    throw std::runtime_error("Invalid state!");
                }
                return std::make_shared<Hyperplane2d>(t[0].cast<bool>(), t[1].cast<xt::xtensor<double, 1>>(), t[2].cast<double>());
            }
        ));
}
