"""Tests for the unified ``intersect_with_obstacles`` entry point.

The unified checker takes one ordered list of mixed Drake ``ConvexSet``
obstacles, dispatches each to the matching kernel (HPolyhedron / hypersphere),
and remaps per-kernel results back to the original list positions. These tests
pin the global-index bookkeeping (the easy thing to get wrong), the ignore map,
type rejection, and agreement with the single-type wrappers.
"""

import numpy as np
import pybezier as pb
import pydrake.all as pd
import pytest

from pybmt.collisions import (
    intersect_with_hpolyhedra,
    intersect_with_hyperspheres,
    intersect_with_obstacles,
)


def _line(p, q, degree=6, t0=0.0, t1=1.0):
    s = np.linspace(0.0, 1.0, degree + 1)[:, None]
    return pb.BezierCurve((1 - s) * np.asarray(p, float) + s * np.asarray(q, float),
                          t0, t1)


def _two_segment_x_axis():
    return pb.CompositeBezierCurve([
        _line([-2, 0], [0, 0], t0=0.0, t1=1.0),
        _line([0, 0], [2, 0], t0=1.0, t1=2.0),
    ])


def _box(lb, ub):
    return pd.HPolyhedron.MakeBox(np.asarray(lb, float), np.asarray(ub, float))


def test_mixed_obstacles_use_global_indices():
    # Interleave types so local (per-kernel) indices differ from global indices:
    #   0: sphere, far       (no hit)
    #   1: box,    on origin  (hits both segments)
    #   2: sphere, on origin  (hits both segments)
    #   3: box,    far        (no hit)
    curve = _two_segment_x_axis()
    obstacles = [
        pd.Hyperellipsoid.MakeHypersphere(0.1, np.array([0.0, 5.0])),
        _box([-0.5, -0.5], [0.5, 0.5]),
        pd.Hyperellipsoid.MakeHypersphere(0.6, np.array([0.0, 0.5])),
        _box([5.0, 5.0], [6.0, 6.0]),
    ]
    hits = intersect_with_obstacles(curve, obstacles, tol=1e-3)
    assert hits == {0: [1, 2], 1: [1, 2]}


def test_empty_and_clear():
    curve = _two_segment_x_axis()
    assert intersect_with_obstacles(curve, []) == {}
    far = [
        pd.Hyperellipsoid.MakeHypersphere(0.1, np.array([0.0, 5.0])),
        _box([5.0, 5.0], [6.0, 6.0]),
    ]
    assert intersect_with_obstacles(curve, far, tol=1e-3) == {}


def test_ignore_uses_global_indices():
    curve = _two_segment_x_axis()
    obstacles = [
        pd.Hyperellipsoid.MakeHypersphere(0.1, np.array([0.0, 5.0])),  # 0
        _box([-0.5, -0.5], [0.5, 0.5]),                                # 1 (box hit)
        pd.Hyperellipsoid.MakeHypersphere(0.6, np.array([0.0, 0.5])),  # 2 (sphere hit)
    ]
    # Ignore the sphere (global idx 2) on both segments; the box (1) remains.
    hits = intersect_with_obstacles(
        curve, obstacles, ignore={0: [2], 1: [2]}, tol=1e-3)
    assert hits == {0: [1], 1: [1]}


def test_unsupported_type_raises():
    curve = _two_segment_x_axis()
    point = pd.Point(np.array([0.0, 0.0]))
    with pytest.raises(TypeError, match="unsupported type"):
        intersect_with_obstacles(curve, [point])


def test_matches_single_type_wrappers():
    curve = _two_segment_x_axis()
    boxes = [_box([-0.5, -0.5], [0.5, 0.5]), _box([5.0, 5.0], [6.0, 6.0])]
    spheres = [
        pd.Hyperellipsoid.MakeHypersphere(0.6, np.array([0.0, 0.5])),
        pd.Hyperellipsoid.MakeHypersphere(0.1, np.array([0.0, 5.0])),
    ]
    # All boxes first, then all spheres -> global indices = local + offset.
    unified = intersect_with_obstacles(curve, boxes + spheres, tol=1e-3)

    h = intersect_with_hpolyhedra(curve, boxes, tol=1e-3)
    s = intersect_with_hyperspheres(curve, spheres, tol=1e-3)
    expected: dict[int, list[int]] = {}
    for seg, obs in h.items():
        expected.setdefault(seg, []).extend(obs)
    for seg, obs in s.items():
        expected.setdefault(seg, []).extend(o + len(boxes) for o in obs)
    expected = {seg: sorted(o) for seg, o in expected.items()}
    assert unified == expected
