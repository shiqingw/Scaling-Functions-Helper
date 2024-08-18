cmake_minimum_required(VERSION 3.18..3.20)

project(scalingFunctionsHelper VERSION 0.1.0)

# Add the cmake directory to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(OpenBLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED CONFIG)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(xtensor-python REQUIRED)
find_package(xsimd REQUIRED)

option(BUILD_PYTHON_MODULE "Build the python model" OFF)

if (BUILD_PYTHON_MODULE)
    pybind11_add_module(
        scalingFunctionsHelperPy
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
    target_link_libraries(scalingFunctionsHelperPy PUBLIC
        pybind11::module
        xtensor
        xtensor::optimize
        # xsimd # Do not use this, code gets slower
        # xtensor::use_xsimd # Do not use this, code gets slower
        xtensor-python
        Python::NumPy
        ${OpenBLAS_LIB}
        LAPACK::LAPACK
        pthread
        m
        dl
    )

    target_compile_definitions(scalingFunctionsHelperPy PRIVATE VERSION_INFO=0.1.0)

    set_property(TARGET scalingFunctionsHelperPy PROPERTY CXX_STANDARD 14)

else()
    # Create a separate library target for your C++ code
    add_library(scalingFunctionsHelper SHARED
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

    # Link the libraries needed by your C++ code
    target_link_libraries(scalingFunctionsHelper PUBLIC
    xtensor
    xtensor::optimize
    # xsimd # Do not use this, code gets slower
    # xtensor::use_xsimd # Do not use this, code gets slower
    xtensor-python
    ${OpenBLAS_LIB}
    LAPACK::LAPACK
    pthread
    m
    dl
    )

    target_compile_definitions(scalingFunctionsHelper PRIVATE VERSION_INFO=0.1.0)

    set_property(TARGET scalingFunctionsHelper PROPERTY CXX_STANDARD 14)

    # Specify the installation paths for the dynamic library
    install(TARGETS scalingFunctionsHelper
        EXPORT scalingFunctionsHelperTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

    # Optionally, install headers
    install(FILES
        src/rimonMethod.hpp
        src/diffOptHelper.hpp
        src/smoothMinimum.hpp
        src/scalingFunction3d.hpp
        src/ellipsoid3d.hpp
        src/logSumExp3d.hpp
        src/hyperplane3d.hpp
        src/superquadrics3d.hpp
        src/scalingFunction2d.hpp
        src/ellipsoid2d.hpp
        src/logSumExp2d.hpp
        src/hyperplane2d.hpp
        DESTINATION include/scalingFunctionsHelper
    )

    # Create and install the package configuration files
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/scalingFunctionsHelperConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
    )

    configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/scalingFunctionsHelperConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/scalingFunctionsHelperConfig.cmake"
    INSTALL_DESTINATION lib/cmake/scalingFunctionsHelper
    )

    install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/scalingFunctionsHelperConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/scalingFunctionsHelperConfigVersion.cmake"
    DESTINATION lib/cmake/scalingFunctionsHelper
    )

    install(EXPORT scalingFunctionsHelperTargets
    FILE scalingFunctionsHelperTargets.cmake
    NAMESPACE scalingFunctionsHelper::
    DESTINATION lib/cmake/scalingFunctionsHelper
    )

endif()
