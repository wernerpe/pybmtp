"""Tests for the polygonal warm start.

The oracle is physical: the warm start must pass through its waypoints, respect
the velocity/acceleration limits along the whole curve, rest at every waypoint,
and be time-contiguous. We verify by dense sampling of the resulting curves.
"""

import numpy as np
import pybezier as pb
import pydrake.all as pd
import pytest

from pybmt import Limits, polygonal_initialization


def box(half_extent, dim):
    """Symmetric box {x : -h <= x_i <= h} as an HPolyhedron."""
    h = np.full(dim, half_extent, dtype=float)
    A = np.vstack([np.eye(dim), -np.eye(dim)])
    b = np.concatenate([h, h])
    return pd.HPolyhedron(A, b)


DOMAIN = box(10.0, 2)


@pytest.fixture
def limits_2d():
    return Limits(box(1.0, 2), box(2.0, 2))  # |v| box 1.0, |a| box 2.0


def sample(curve, n=400):
    ts = np.linspace(curve.initial_time, curve.final_time, n)
    return ts, np.array([curve(t) for t in ts])


def test_passes_waypoints_and_time_contiguous(limits_2d):
    wps = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    traj = polygonal_initialization(wps, DOMAIN, limits_2d, degree=6)
    assert isinstance(traj, pb.CompositeBezierCurve)
    assert len(traj.curves) == 3
    np.testing.assert_allclose(traj(traj.initial_time), wps[0], atol=1e-6)
    np.testing.assert_allclose(traj(traj.final_time), wps[-1], atol=1e-6)
    for c0, c1 in zip(traj.curves[:-1], traj.curves[1:]):
        assert c0.final_time == pytest.approx(c1.initial_time)
        np.testing.assert_allclose(c0(c0.final_time), c1(c1.initial_time), atol=1e-6)


def test_respects_velocity_and_acceleration(limits_2d):
    traj = polygonal_initialization(np.array([[0.0, 0.0], [2.0, 1.0]]), DOMAIN, limits_2d, degree=6)
    vdot = traj.derivative()
    vddot = vdot.derivative()
    _, vs = sample(vdot)
    _, as_ = sample(vddot)
    assert np.max(np.abs(vs)) <= 1.0 + 1e-3
    assert np.max(np.abs(as_)) <= 2.0 + 1e-2


def test_rests_at_every_waypoint(limits_2d):
    wps = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    traj = polygonal_initialization(wps, DOMAIN, limits_2d, degree=6)
    vdot = traj.derivative()
    for c in traj.curves:  # each segment starts and ends at rest
        np.testing.assert_allclose(vdot(c.initial_time), [0, 0], atol=1e-6)
        np.testing.assert_allclose(vdot(c.final_time), [0, 0], atol=1e-6)


def test_force_same_segment_time(limits_2d):
    # Segments of clearly different length -> different natural durations.
    wps = np.array([[0.0, 0.0], [0.2, 0.0], [3.0, 0.0]])
    traj = polygonal_initialization(wps, DOMAIN, limits_2d, degree=6, force_same_segment_time=True)
    durations = [c.final_time - c.initial_time for c in traj.curves]
    assert durations[0] == pytest.approx(durations[1], rel=1e-9)


def test_velocity_only_warm_start():
    # J=1: a velocity-only limit still yields a valid rest-to-rest warm start.
    traj = polygonal_initialization(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]]), DOMAIN, Limits(box(1.0, 2)), degree=6
    )
    assert len(traj.curves) == 2
    np.testing.assert_allclose(traj(traj.initial_time), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(traj(traj.final_time), [2.0, 1.0], atol=1e-6)


def test_stays_on_each_waypoint_segment(limits_2d):
    # Every segment must lie exactly on the straight line between its waypoints and
    # stay within it (progress in [0, 1]), so a collision-free polyline yields a
    # collision-free seed. A skew segment is the interesting case (its free min-time
    # curve would otherwise bow slightly off the line).
    wps = np.array([[0.0, 0.0], [2.0, 1.0], [1.0, 3.0]])
    traj = polygonal_initialization(wps, DOMAIN, limits_2d, degree=6)
    for seg, a, b in zip(traj.curves, wps[:-1], wps[1:]):
        ts = np.linspace(seg.initial_time, seg.final_time, 200)
        xy = np.array([seg(t) for t in ts])
        u = (b - a) / np.linalg.norm(b - a)
        perp = xy - a - np.outer((xy - a) @ u, u)
        progress = ((xy - a) @ u) / np.linalg.norm(b - a)
        assert np.max(np.linalg.norm(perp, axis=1)) < 1e-9  # on the line
        assert progress.min() > -1e-9 and progress.max() < 1 + 1e-9  # within the segment
