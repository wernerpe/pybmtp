"""Tests for the Python collision-checking wrapper over the native kernel.

These mirror the C++ gtest cases at the Python boundary: they verify that
pybezier curves round-trip into the native kernel correctly and that the
conservative guarantee (no false "collision-free") holds on sampled points.
"""

import numpy as np
import pybezier as pb
import pydrake.all as pd
import pytest

from pybmt.collisions import (
    curve_is_collision_free,
    intersect_with_hpolyhedra,
    to_native_composite,
)


def box(lb, ub):
    """Axis-aligned box {x : lb <= x <= ub} as an HPolyhedron."""
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    d = lb.size
    A = np.vstack([np.eye(d), -np.eye(d)])
    b = np.concatenate([ub, -lb])
    return pd.HPolyhedron(A, b)


def straight_line(start, end, degree, t0=0.0, t1=1.0):
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    s = np.linspace(0.0, 1.0, degree + 1)[:, None]
    pts = (1.0 - s) * start[None, :] + s * end[None, :]
    return pb.BezierCurve(pts, t0, t1)


def test_far_curve_is_collision_free():
    curve = straight_line([-2, 0], [2, 0], 6)
    obstacle = box([-1, 4], [1, 6])  # box up near y=5
    assert curve_is_collision_free(curve, obstacle, tol=1e-3)


def test_curve_through_obstacle_collides():
    curve = straight_line([-2, 0], [2, 0], 6)
    obstacle = box([-1, -1], [1, 1])  # box at origin
    assert not curve_is_collision_free(curve, obstacle, tol=1e-3)


def test_never_false_positive_on_sampled_curve():
    obstacle = box([-0.3, -0.3], [0.3, 0.3])
    curve = straight_line([-2, 0.4], [2, 0.4], 6)
    assert curve_is_collision_free(curve, obstacle, tol=1e-3)
    A, b = np.asarray(obstacle.A()), np.asarray(obstacle.b())
    for t in np.linspace(0.0, 1.0, 1001):
        x = curve(t)
        assert (A @ x - b).max() >= 0.0, f"sampled point inside obstacle at t={t}"


def test_composite_maps_segments_and_honors_ignore():
    seg0 = straight_line([-2, 0], [0, 0], 6, 0.0, 1.0)
    seg1 = straight_line([0, 0], [2, 4], 6, 1.0, 2.0)
    curve = pb.CompositeBezierCurve([seg0, seg1])

    obs = [
        box([-1.2, -0.5], [-0.8, 0.5]),  # sits on seg0
        box([5, 5], [6, 6]),             # far away
    ]

    hits = intersect_with_hpolyhedra(curve, obs, tol=1e-3, parallelize=False)
    assert hits == {0: [0]}

    hits_ignored = intersect_with_hpolyhedra(
        curve, obs, ignore={0: [0]}, tol=1e-3, parallelize=False)
    assert hits_ignored == {}


def test_parallel_matches_serial():
    rng = np.random.default_rng(0)
    segs = [straight_line([s, 0.0], [s + 1, 0.0], 6, s, s + 1) for s in range(64)]
    curve = pb.CompositeBezierCurve(segs)
    obs = []
    for _ in range(200):
        c = np.array([rng.uniform(-1, 1) * 64.0, rng.uniform(-1, 1) * 0.05])
        obs.append(box(c - 0.02, c + 0.02))
    serial = intersect_with_hpolyhedra(curve, obs, tol=1e-5, parallelize=False)
    parallel = intersect_with_hpolyhedra(curve, obs, tol=1e-5, parallelize=True)
    assert serial == parallel


def test_to_native_composite_preserves_geometry():
    seg0 = straight_line([0, 0], [1, 1], 6, 0.0, 1.0)
    seg1 = straight_line([1, 1], [2, 0], 6, 1.0, 2.0)
    curve = pb.CompositeBezierCurve([seg0, seg1])
    native = to_native_composite(curve)
    assert len(native) == 2
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        np.testing.assert_allclose(native(t), curve(t), atol=1e-12)
