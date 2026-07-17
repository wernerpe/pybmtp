"""Tests for the separating-plane LP and the conic supporting-plane geometry.

Oracle: a correct separating plane (a(t), b(t)) must put the obstacle on its
non-negative side (a.x + b >= 0 for every obstacle point) and the trajectory
segment on its non-positive side (a.r + b <= 0 along the segment). We verify
both by sampling.
"""

import numpy as np
import pybezier as pb
import pydrake.all as pd
import pytest

from pybmtp.geometry import add_supporting_plane, get_conic_description
from pybmtp.updaters import PlaneUpdater


def box(lb, ub):
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    d = lb.size
    A = np.vstack([np.eye(d), -np.eye(d)])
    return pd.HPolyhedron(A, np.concatenate([ub, -lb]))


def straight_segment(start, end, degree, t0, t1):
    start, end = np.asarray(start, float), np.asarray(end, float)
    s = np.linspace(0, 1, degree + 1)[:, None]
    return pb.BezierCurve((1 - s) * start[None] + s * end[None], t0, t1)


def box_vertices(lb, ub):
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    from itertools import product

    return np.array(list(product(*zip(lb, ub))))


# --------------------------------------------------------------- geometry unit


def test_supporting_plane_separates_box_from_point():
    obstacle = box([-1, -1], [1, 1])
    prog = pd.MathematicalProgram()
    a, b = add_supporting_plane(prog, obstacle, "t", step_back=1e-6)
    # Force the plane to also keep the external point (0, 3) on the negative
    # side: a.p + b <= 0. A feasible solution then strictly separates them.
    p = np.array([0.0, 3.0])
    prog.AddLinearConstraint(a @ p + b[0] <= 0)
    result = pd.ClarabelSolver().Solve(prog)
    assert result.is_success()
    a_s, b_s = result.GetSolution(a), result.GetSolution(b)[0]
    for v in box_vertices([-1, -1], [1, 1]):
        assert a_s @ v + b_s >= -1e-6
    assert a_s @ p + b_s <= 1e-6


def test_conic_description_types():
    A, B, c, cone = get_conic_description(box([-1, -1], [1, 1]))
    assert cone == "nonneg_orthant" and B is None
    A, B, c, cone = get_conic_description(pd.Hyperellipsoid(np.eye(2), np.zeros(2)))
    assert cone == "soc"


# --------------------------------------------------------------- plane updater


@pytest.fixture
def updater_problem():
    obstacle = box([-1, -1], [1, 1])  # box at origin
    # Single-segment trajectory passing well above the box.
    seg = straight_segment([-2, 2], [2, 2], 6, 0.0, 1.0)
    traj = pb.CompositeBezierCurve([seg])
    return obstacle, traj


def test_plane_supports_obstacle_and_separates_trajectory(updater_problem):
    obstacle, traj = updater_problem
    upd = PlaneUpdater([obstacle], plane_degree=1)
    planes = upd.compute_planes({0: [0]}, traj)
    assert (0, 0) in planes
    a_curve, b_curve = planes[(0, 0)]

    # Obstacle side: a(t).x + b(t) >= 0 for box vertices at sampled t.
    for t in np.linspace(traj.initial_time, traj.final_time, 25):
        a, b = a_curve(t), b_curve(t)[0]
        for v in box_vertices([-1, -1], [1, 1]):
            assert a @ v + b >= -1e-6

    # Trajectory side: a(t).r(t) + b(t) <= 0 along the segment.
    seg = traj.curves[0]
    for t in np.linspace(seg.initial_time, seg.final_time, 50):
        a, b = a_curve(t), b_curve(t)[0]
        assert a @ seg(t) + b <= 1e-6


def test_no_tagged_obstacles_returns_empty(updater_problem):
    obstacle, traj = updater_problem
    upd = PlaneUpdater([obstacle], plane_degree=1)
    assert upd.compute_planes({}, traj) == {}


def test_base_program_is_cached(updater_problem):
    obstacle, traj = updater_problem
    upd = PlaneUpdater([obstacle], plane_degree=1)
    upd.compute_planes({0: [0]}, traj)
    assert 0 in upd._base_progs
    cached = upd._base_progs[0]
    upd.compute_planes({0: [0]}, traj)
    assert upd._base_progs[0] is cached  # not rebuilt


def test_separates_against_hyperellipsoid():
    obstacle = pd.Hyperellipsoid(np.eye(2), np.zeros(2))  # unit disk at origin
    seg = straight_segment([-2, 2], [2, 2], 6, 0.0, 1.0)
    traj = pb.CompositeBezierCurve([seg])
    planes = PlaneUpdater([obstacle], plane_degree=1).compute_planes({0: [0]}, traj)
    a_curve, b_curve = planes[(0, 0)]
    # Disk side: a.x + b >= 0 for points on the unit circle.
    for t in np.linspace(0, 1, 20):
        a, b = a_curve(t), b_curve(t)[0]
        for theta in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            x = np.array([np.cos(theta), np.sin(theta)])
            assert a @ x + b >= -1e-6
