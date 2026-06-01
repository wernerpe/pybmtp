"""Tests for the trajectory-update SOCP.

The central oracle is the equivalence the dual-form comments claim: the fast
matrix encoding and the readable symbolic encoding must define the *same*
optimization problem, hence produce the same trajectory and segment time. We
also check the physical properties of the solution (boundary conditions,
velocity/acceleration limits) and that a separating plane is honored.
"""

import numpy as np
import pybezier as pb
import pydrake.all as pd
import pytest

from pybmt.updaters import TrajectoryUpdater


def box(half, dim):
    h = np.full(dim, half, float)
    A = np.vstack([np.eye(dim), -np.eye(dim)])
    return pd.HPolyhedron(A, np.concatenate([h, h]))


@pytest.fixture
def problem():
    dim = 2
    return dict(
        source=np.array([0.0, 0.0]),
        target=np.array([1.0, 0.5]),
        domain=box(5.0, dim),
        num_segments=3,
        velocity_set=box(1.0, dim),
        acceleration_set=box(2.0, dim),
        degree=6,
    )


def make_updater(problem, **kw):
    return TrajectoryUpdater(
        problem["source"], problem["target"], problem["domain"],
        problem["num_segments"], problem["velocity_set"],
        problem["acceleration_set"], degree=problem["degree"], **kw)


def sample(curve, n=400):
    ts = np.linspace(curve.initial_time, curve.final_time, n)
    return np.array([curve(t) for t in ts])


def test_matrix_solves_and_respects_boundaries(problem):
    upd = make_updater(problem)
    traj, seg_time, cost, _ = upd.solve({})
    assert seg_time > 0
    np.testing.assert_allclose(traj(traj.initial_time), problem["source"], atol=1e-6)
    np.testing.assert_allclose(traj(traj.final_time), problem["target"], atol=1e-6)


def test_matrix_respects_velocity_acceleration(problem):
    upd = make_updater(problem)
    traj, seg_time, _, _ = upd.solve({})
    # The returned trajectory spans normalized time [0, 1] over N segments, so
    # its derivatives are in normalized time. Physical total duration is
    # N * seg_time, hence physical velocity = d/du * du/dt = derivative /
    # (N * seg_time), and acceleration scales by the square.
    n = problem["num_segments"]
    vel = traj.derivative()
    acc = vel.derivative()
    scale_v = 1.0 / (n * seg_time)
    scale_a = 1.0 / (n * seg_time) ** 2
    assert np.max(np.abs(sample(vel))) * scale_v <= 1.0 + 1e-3
    assert np.max(np.abs(sample(acc))) * scale_a <= 2.0 + 1e-2


def test_symbolic_and_matrix_agree(problem):
    matrix = make_updater(problem, use_symbolic_constraints=False)
    symbolic = make_updater(problem, use_symbolic_constraints=True)
    assert matrix.use_symbolic_constraints is False
    assert symbolic.use_symbolic_constraints is True

    tm, st_m, cost_m, _ = matrix.solve({})
    ts, st_s, cost_s, _ = symbolic.solve({})

    # The two encodings define the same program, so the objective (segment time
    # / cost) must match tightly. Trajectory positions can differ by solver
    # tolerance because the spatial optimum is mildly degenerate.
    assert st_m == pytest.approx(st_s, rel=1e-4)
    assert cost_m == pytest.approx(cost_s, rel=1e-4)
    grid = np.linspace(tm.initial_time, tm.final_time, 200)
    for t in grid:
        np.testing.assert_allclose(tm(t), ts(t), atol=2e-3)


def test_plane_constraint_is_honored_both_forms(problem):
    # A plane that forbids y > 0.1 over segment 1: a(t) = (0, 1), b(t) = -0.1
    # encodes a(t).r(t) + b(t) = y - 0.1 <= 0.
    a_curve = pb.BezierCurve(np.array([[0.0, 1.0], [0.0, 1.0]]), 0, 1)
    b_curve = pb.BezierCurve(np.array([[-0.1], [-0.1]]), 0, 1)
    seg = 1
    for symbolic in (False, True):
        upd = make_updater(problem, use_symbolic_constraints=symbolic)
        traj, _, _, _ = upd.solve({(seg, 0): (a_curve, b_curve)})
        ys = sample(traj.curves[seg])[:, 1]
        assert ys.max() <= 0.1 + 1e-6, f"plane violated (symbolic={symbolic})"


def test_clear_planes_restores_unconstrained_optimum(problem):
    upd = make_updater(problem)
    base_traj, base_time, _, _ = upd.solve({})
    a_curve = pb.BezierCurve(np.array([[0.0, 1.0], [0.0, 1.0]]), 0, 1)
    b_curve = pb.BezierCurve(np.array([[-0.1], [-0.1]]), 0, 1)
    constrained_traj, constrained_time, _, _ = upd.solve({(1, 0): (a_curve, b_curve)})
    # The added plane cannot make the problem faster.
    assert constrained_time >= base_time - 1e-6
    upd.clear_planes()
    _, restored_time, _, _ = upd.solve({})
    assert restored_time == pytest.approx(base_time, rel=1e-5)
