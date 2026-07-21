cmake_minimum_required(VERSION 3.15)
project(mat_neuron LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

# Prefer a system Eigen3 install (this is what cibuildwheel's
# yum/brew/apk before-all steps provide in CI)
find_package(Eigen3 3.3 QUIET NO_MODULE)

if(NOT TARGET Eigen3::Eigen)
    # Fall back to a vendored copy or well-known local install paths
    find_path(EIGEN3_INCLUDE_DIR
        NAMES Eigen/Dense
        PATHS
            "${CMAKE_CURRENT_SOURCE_DIR}/include/eigen"
            /usr/include/eigen3
            /usr/local/include/eigen3
            /opt/local/include/eigen3
    )
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Eigen3 not found. Install it system-wide or "
                             "vendor it at include/eigen.")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

pybind11_add_module(_model MODULE mat_neuron/_model.cpp)
target_link_libraries(_model PRIVATE Eigen3::Eigen)

install(TARGETS _model DESTINATION mat_neuron)
