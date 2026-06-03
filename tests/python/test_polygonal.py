"""Tests for the PolygonalInitializer warm start.

The oracle here is physical: a minimum-time, constraint-feasible segment must
(a) actually pass through its waypoints, (b) never exceed the velocity or
acceleration box anywhere along the curve, and (c) start/end at rest. We verify
these by dense sampling of the resulting Bezier curves.
"""

import numpy as np
import pybezier as pb
import pydrake.all as pd
import pytest

from pybmt import PolygonalInitializer
from pybmt.polygonal import compute_set_limit, solve_segment_time_optimal


def box(half_extent, dim):
    """Symmetric box {x : -h <= x_i <= h} as an HPolyhedron."""
    h = np.full(dim, half_extent, dtype=float)
    A = np.vstack([np.eye(dim), -np.eye(dim)])
    b = np.concatenate([h, h])
    return pd.HPolyhedron(A, b)


@pytest.fixture
def vel_acc_2d():
    return box(1.0, 2), box(2.0, 2)  # |v| box 1.0, |a| box 2.0


def sample(curve, n=400):
    ts = np.linspace(curve.initial_time, curve.final_time, n)
    return ts, np.array([curve(t) for t in ts])


def test_compute_set_limit_hpolyhedron_axis():
    # Unit box, direction +x -> limit is 1.0 (face at x=1).
    s = box(1.0, 3)
    assert compute_set_limit(np.array([1.0, 0.0, 0.0]), s) == pytest.approx(1.0)
    # Diagonal direction -> hits x=1 face first at alpha = 1/component.
    d = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    assert compute_set_limit(d, s) == pytest.approx(np.sqrt(2), rel=1e-9)


def test_single_segment_hits_endpoints(vel_acc_2d):
    v, a = vel_acc_2d
    start, end = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    curve = solve_segment_time_optimal(start, end, v, a, degree=6)
    assert curve is not None
    np.testing.assert_allclose(curve(curve.initial_time), start, atol=1e-6)
    np.testing.assert_allclose(curve(curve.final_time), end, atol=1e-6)


def test_segment_respects_velocity_and_acceleration(vel_acc_2d):
    v, a = vel_acc_2d
    curve = solve_segment_time_optimal(np.array([0.0, 0.0]), np.array([2.0, 1.0]), v, a, degree=6)
    vdot = curve.derivative()
    vddot = vdot.derivative()
    _, vs = sample(vdot)
    _, as_ = sample(vddot)
    # Small tolerance for sampling/numerical slack.
    assert np.max(np.abs(vs)) <= 1.0 + 1e-3
    assert np.max(np.abs(as_)) <= 2.0 + 1e-2


def test_segment_starts_and_ends_at_rest(vel_acc_2d):
    v, a = vel_acc_2d
    curve = solve_segment_time_optimal(np.array([0.0, 0.0]), np.array([1.0, 1.0]), v, a, degree=6)
    vdot = curve.derivative()
    np.testing.assert_allclose(vdot(curve.initial_time), [0, 0], atol=1e-6)
    np.testing.assert_allclose(vdot(curve.final_time), [0, 0], atol=1e-6)


def test_initializer_is_time_contiguous_and_passes_waypoints(vel_acc_2d):
    v, a = vel_acc_2d
    wps = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    init = PolygonalInitializer(v, a, trajectory_degree=6)
    traj = init.initialize(wps)
    assert isinstance(traj, pb.CompositeBezierCurve)
    assert len(traj.curves) == 3
    # Each waypoint is reached at the corresponding transition time.
    np.testing.assert_allclose(traj(traj.initial_time), wps[0], atol=1e-6)
    np.testing.assert_allclose(traj(traj.final_time), wps[-1], atol=1e-6)
    # Segments are contiguous in time.
    for c0, c1 in zip(traj.curves[:-1], traj.curves[1:]):
        assert c0.final_time == pytest.approx(c1.initial_time)


def test_force_same_segment_time(vel_acc_2d):
    v, a = vel_acc_2d
    # Segments of clearly different length -> different natural durations.
    wps = np.array([[0.0, 0.0], [0.2, 0.0], [3.0, 0.0]])
    traj = PolygonalInitializer(v, a, trajectory_degree=6, force_same_segment_time=True).initialize(
        wps
    )
    durations = [c.final_time - c.initial_time for c in traj.curves]
    assert durations[0] == pytest.approx(durations[1], rel=1e-9)
