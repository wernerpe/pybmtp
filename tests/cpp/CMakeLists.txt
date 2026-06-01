# GoogleTest target for the native Bezier collision kernel.
# Fetched at configure time so the test build needs no system gtest.
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_bezier_collision test_bezier_collision.cpp
               $<TARGET_OBJECTS:pybmt_bezier>)
target_include_directories(test_bezier_collision PRIVATE ${CMAKE_SOURCE_DIR}/cpp/include)
target_link_libraries(test_bezier_collision PRIVATE Eigen3::Eigen GTest::gtest_main)
if(OpenMP_CXX_FOUND)
  target_link_libraries(test_bezier_collision PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(test_bezier_collision PRIVATE PYBMT_HAVE_OPENMP=1)
else()
  target_compile_definitions(test_bezier_collision PRIVATE PYBMT_HAVE_OPENMP=0)
endif()

include(GoogleTest)
gtest_discover_tests(test_bezier_collision)
