/**
 * pybind11 bindings for pybmtp's native Bezier collision check.
 *
 * Exposes the minimal surface the Python layer needs:
 *   - BezierCurve / CompositeBezierCurve (so we can round-trip from pybezier),
 *   - the single-curve collision predicate (handy for unit tests),
 *   - the parallel composite-curve-vs-HPolyhedra intersector (the hot path used
 *     inside the BCP outer loop).
 *
 * Eigen <-> numpy and STL <-> Python conversions are provided by pybind11's
 * eigen.h / stl.h headers. The module is imported as ``pybmtp._native``.
 */
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybmtp/bezier_curve.h"

namespace py = pybind11;
using namespace pybmtp;

PYBIND11_MODULE(_native, m) {
  m.doc() = "Native Bezier geometry and parallel collision checking for pybmtp.";

  py::class_<BezierCurve>(m, "BezierCurve")
      .def(py::init<const Eigen::MatrixXd&, double, double>(), py::arg("points"),
           py::arg("initial_time") = 0.0, py::arg("final_time") = 1.0,
           "Construct from a (degree+1, dim) control-point matrix and a time "
           "interval.")
      .def_property_readonly("points", &BezierCurve::points)
      .def_property_readonly("degree", &BezierCurve::degree)
      .def_property_readonly("dimension", &BezierCurve::dimension)
      .def_property_readonly("initial_time", &BezierCurve::initial_time)
      .def_property_readonly("final_time", &BezierCurve::final_time)
      .def_property_readonly("duration", &BezierCurve::duration)
      .def("__call__", &BezierCurve::operator(), py::arg("time"))
      .def("derivative", &BezierCurve::derivative)
      .def("domain_split", &BezierCurve::domain_split, py::arg("split_time"));

  py::class_<CompositeBezierCurve>(m, "CompositeBezierCurve")
      .def(py::init<const std::vector<BezierCurve>&>(), py::arg("curves"),
           "Construct from a time-contiguous list of BezierCurve segments.")
      .def_property_readonly("dimension", &CompositeBezierCurve::dimension)
      .def_property_readonly("initial_time",
                             &CompositeBezierCurve::initial_time)
      .def_property_readonly("final_time", &CompositeBezierCurve::final_time)
      .def_property_readonly("duration", &CompositeBezierCurve::duration)
      .def_property_readonly("num_segments",
                             &CompositeBezierCurve::num_segments)
      .def("__call__", &CompositeBezierCurve::operator(), py::arg("time"))
      .def("__len__", &CompositeBezierCurve::num_segments)
      .def("__getitem__", &CompositeBezierCurve::operator[], py::arg("index"),
           py::return_value_policy::reference_internal);

  m.def("bezier_curve_hpolyhedron_collision_free",
        &BezierCurveHPolyhedronCollisionFree, py::arg("curve"), py::arg("A"),
        py::arg("b"), py::arg("tol") = 1e-2,
        "True if the curve is proven to avoid the polytope {x : A x <= b}. "
        "Conservative: never reports a colliding curve as collision-free.");

  m.def("intersect_composite_bezier_with_hpolyhedra",
        &IntersectCompositeBezierCurveWithHPolyhedra, py::arg("curve"),
        py::arg("As"), py::arg("bs"), py::arg("hpoly_to_ignore"),
        py::arg("tol") = 1e-2, py::arg("parallelize") = true,
        "Per-segment obstacle intersection test. Returns {segment_index: "
        "[obstacle_indices]} for segments that are not proven collision-free. "
        "Parallelized over segments with OpenMP when available.");

  m.def("bezier_curve_sphere_collision_free", &BezierCurveSphereCollisionFree,
        py::arg("curve"), py::arg("center"), py::arg("radius"),
        py::arg("tol") = 1e-2,
        "True if the curve is proven to avoid the ball of the given center and "
        "radius. Conservative: never reports a colliding curve as free.");

  m.def("intersect_composite_bezier_with_spheres",
        &IntersectCompositeBezierCurveWithSpheres, py::arg("curve"),
        py::arg("centers"), py::arg("radii"), py::arg("spheres_to_ignore"),
        py::arg("tol") = 1e-2, py::arg("parallelize") = true,
        "Per-segment sphere intersection test. Unlike the HPolyhedra variant, "
        "'spheres_to_ignore' is one ignore-list per segment (length == number "
        "of segments) and the result is a per-segment list of colliding sphere "
        "indices (empty for collision-free segments).");
}
