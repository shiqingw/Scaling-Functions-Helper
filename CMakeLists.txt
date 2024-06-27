cmake_minimum_required(VERSION 3.18..3.20)

project(scalingFunctionsHelper)

find_package(OpenBLAS REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED CONFIG)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(xtensor-python REQUIRED)

pybind11_add_module(
    scalingFunctionsHelper
    src/pybind.cpp
    src/rimonMethod.cpp
    src/diffOptHelper.cpp
    src/smoothMinimum.cpp
    src/scalingFunction3d.cpp
    src/ellipsoid3d.cpp
    src/logSumExp3d.cpp
    src/hyperplane3d.cpp
    src/superquadrics3d.cpp
    src/scalingFunction2d.cpp
    src/ellipsoid2d.cpp
    src/logSumExp2d.cpp
    src/hyperplane2d.cpp
)

# If OpenBLAS provides a target, use it directly. If not, specify the library path as you did.
target_link_libraries(scalingFunctionsHelper PUBLIC
    pybind11::module
    xtensor-python
    Python::NumPy
    ${OpenBLAS_LIBRARIES}  # Use the variable set by find_package if available
    pthread
    m
    dl
)

target_include_directories(scalingFunctionsHelper PUBLIC
    ${OpenBLAS_INCLUDE_DIRS}  # Include this if find_package sets this variable
)

target_compile_definitions(scalingFunctionsHelper PRIVATE VERSION_INFO=0.1.0)

# Specify the installation paths
install(TARGETS scalingFunctionsHelper
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

# Optionally, install headers
install(FILES
        src/scalingFunction3d.hpp
        src/ellipsoid3d.hpp
        src/logSumExp3d.hpp
        src/hyperplane3d.hpp
        src/superquadrics3d.hpp
        src/scalingFunction2d.hpp
        src/ellipsoid2d.hpp
        src/logSumExp2d.hpp
        src/hyperplane2d.hpp
        DESTINATION include/scalingFunctionsHelper)