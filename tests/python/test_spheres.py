"""Tests for the sphere (ball) collision wrapper ``intersect_with_hyperspheres``.

The native sphere kernel was vendored from csdecomp alongside the HPolyhedron
path but was never exposed to Python. These tests pin down the Python wrapper:
correctness on free/colliding scenes, Drake ``Hyperellipsoid`` acceptance (with
anisotropic ellipsoids rejected), the ignore mask, the never-false-positive
guarantee, and parallel/serial agreement.
"""

import numpy as np
import pybezier as pb
import pydrake.all as pd
import pytest

from pybmt.collisions import intersect_with_hyperspheres


def _line(p, q, degree=6, t0=0.0, t1=1.0):
    """Degree-``degree`` Bezier whose control points trace the segment p->q."""
    s = np.linspace(0.0, 1.0, degree + 1)[:, None]
    return pb.BezierCurve((1 - s) * np.asarray(p, float) + s * np.asarray(q, float),
                          t0, t1)


def _two_segment_x_axis():
    """Composite curve along the x-axis (y=0) from x=-2 to x=2."""
    return pb.CompositeBezierCurve([
        _line([-2, 0], [0, 0], t0=0.0, t1=1.0),
        _line([0, 0], [2, 0], t0=1.0, t1=2.0),
    ])


def test_far_sphere_is_collision_free():
    curve = _two_segment_x_axis()
    sphere = pd.Hyperellipsoid.MakeHypersphere(0.1, np.array([0.0, 0.5]))
    assert intersect_with_hyperspheres(curve, [sphere], tol=1e-3) == {}


def test_sphere_on_path_collides_both_segments():
    curve = _two_segment_x_axis()
    # Big ball straddling the origin: both segments pass through it.
    sphere = pd.Hyperellipsoid.MakeHypersphere(0.6, np.array([0.0, 0.5]))
    assert intersect_with_hyperspheres(curve, [sphere], tol=1e-3) == {0: [0], 1: [0]}


def test_hypersphere_matches_raw_center_radius():
    curve = _two_segment_x_axis()
    c, r = np.array([0.0, 0.5]), 0.6
    drake = intersect_with_hyperspheres(
        curve, [pd.Hyperellipsoid.MakeHypersphere(r, c)], tol=1e-3)
    raw = intersect_with_hyperspheres(curve, [(c, r)], tol=1e-3)
    assert drake == raw == {0: [0], 1: [0]}


def test_anisotropic_ellipsoid_is_rejected():
    curve = _two_segment_x_axis()
    ellipsoid = pd.Hyperellipsoid(np.diag([1.0, 2.0]), np.zeros(2))
    with pytest.raises(ValueError, match="isotropic"):
        intersect_with_hyperspheres(curve, [ellipsoid])


def test_ignore_mask_suppresses_hit():
    curve = _two_segment_x_axis()
    sphere = pd.Hyperellipsoid.MakeHypersphere(0.6, np.array([0.0, 0.5]))
    hits = intersect_with_hyperspheres(
        curve, [sphere], ignore={0: [0], 1: [0]}, tol=1e-3)
    assert hits == {}


def test_never_false_positive_on_sampled_curve():
    # A curve reported collision-free must have no densely sampled point inside
    # any sphere.
    curve = _two_segment_x_axis()
    centers = [np.array([x, 0.5]) for x in (-1.0, 0.0, 1.0)]
    spheres = [pd.Hyperellipsoid.MakeHypersphere(0.3, c) for c in centers]
    hits = intersect_with_hyperspheres(curve, spheres, tol=1e-3)
    assert hits == {}, f"expected clear, got {hits}"

    for k in range(2001):
        t = curve.initial_time + (curve.final_time - curve.initial_time) * k / 2000
        x = np.asarray(curve(t), float)
        for c in centers:
            assert np.linalg.norm(x - c) > 0.3 - 1e-9, f"point inside sphere at t={t}"


def test_parallel_matches_serial():
    rng = np.random.default_rng(0)
    curve = _two_segment_x_axis()
    spheres = [
        pd.Hyperellipsoid.MakeHypersphere(
            0.05, np.array([rng.uniform(-2, 2), rng.uniform(-0.05, 0.05)]))
        for _ in range(80)
    ]
    serial = intersect_with_hyperspheres(curve, spheres, tol=1e-5, parallelize=False)
    parallel = intersect_with_hyperspheres(curve, spheres, tol=1e-5, parallelize=True)
    assert serial == parallel
