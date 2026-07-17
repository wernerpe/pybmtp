cmake_minimum_required(VERSION 3.18)
project(pybmtp_native LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
#   PYBMTP_BUILD_PYTHON  ON  -> build the pybind11 extension (default; this is
#                              what scikit-build-core invokes for a wheel).
#   PYBMTP_BUILD_TESTS   OFF -> build the C++ gtest target (turned on by the
#                              native-test CI job and for local `ctest`).
option(PYBMTP_BUILD_PYTHON "Build the pybind11 extension module" ON)
option(PYBMTP_BUILD_TESTS  "Build the native gtest targets"      OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# Eigen: prefer a system install; otherwise fetch a pinned release so the
# source build works on a clean machine (e.g. cibuildwheel manylinux).
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
  )
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(Eigen3)
endif()

# OpenMP: optional. Without it the kernel still compiles and runs serially
# (the #pragma omp directives are simply ignored).
find_package(OpenMP QUIET)

# The vendored Bezier geometry + collision kernel, built once as an object
# library and shared between the Python extension and the gtest target.
add_library(pybmtp_bezier OBJECT cpp/src/bezier_curve.cpp)
target_include_directories(pybmtp_bezier PUBLIC cpp/include)
target_link_libraries(pybmtp_bezier PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(pybmtp_bezier PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# Python extension
# ---------------------------------------------------------------------------
if(PYBMTP_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_native cpp/src/bindings.cpp $<TARGET_OBJECTS:pybmtp_bezier>)
  target_include_directories(_native PRIVATE cpp/include)
  target_link_libraries(_native PRIVATE Eigen3::Eigen)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(_native PRIVATE OpenMP::OpenMP_CXX)
  endif()

  # Install the compiled module into the package directory inside the wheel.
  install(TARGETS _native DESTINATION pybmtp)
endif()

# ---------------------------------------------------------------------------
# Native tests
# ---------------------------------------------------------------------------
if(PYBMTP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
