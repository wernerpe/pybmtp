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

from pybmtp.limits import Limits
from pybmtp.updaters import TrajectoryUpdater


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
    limits = kw.pop("limits", Limits(problem["velocity_set"], problem["acceleration_set"]))
    return TrajectoryUpdater(
        problem["source"],
        problem["target"],
        problem["domain"],
        problem["num_segments"],
        limits,
        degree=kw.pop("degree", problem["degree"]),
        **kw,
    )


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


# --------------------------------------------------------------- J = 3, 4 (jerk/snap)


def test_acceleration_only_ladder_is_minimal(problem):
    upd = make_updater(problem)
    assert upd.J == 2
    assert len(upd.T_powers) == 3  # [1, T, T**2] -- no larger than acceleration needs
    assert upd.terminal_order == 2  # default matches J: r'(0)=r''(0)=0


def test_jerk_snap_grow_ladder_and_respect_limits(problem):
    limits = Limits(box(1.0, 2), box(2.0, 2), jerk=box(6.0, 2), snap=box(60.0, 2))
    upd = TrajectoryUpdater(
        problem["source"],
        problem["target"],
        problem["domain"],
        problem["num_segments"],
        limits,
        degree=9,
    )
    assert upd.J == 4
    assert len(upd.T_powers) == 5  # [1, T, T**2, T**3, T**4]
    traj, seg_time, _, _ = upd.solve({})
    n = problem["num_segments"]
    v = traj.derivative()
    a = v.derivative()
    j = a.derivative()
    s = j.derivative()
    sc = 1.0 / (n * seg_time)
    assert np.max(np.abs(sample(v))) * sc <= 1.0 + 1e-2
    assert np.max(np.abs(sample(a))) * sc**2 <= 2.0 + 1e-2
    assert np.max(np.abs(sample(j))) * sc**3 <= 6.0 + 1e-1
    assert np.max(np.abs(sample(s))) * sc**4 <= 60.0 + 1.0


def test_snap_respected_with_partial_terminal_rest(problem):
    # J=4 but only velocity+acceleration rest at the ends (terminal_order=2): the
    # jerk/snap boundary control points are NOT pinned, so they must still be
    # constrained. Regression for the boundary-skip bug (snap escaping its bound
    # at the endpoints).
    limits = Limits(box(1.0, 2), box(2.0, 2), jerk=box(6.0, 2), snap=box(60.0, 2))
    upd = TrajectoryUpdater(
        problem["source"],
        problem["target"],
        problem["domain"],
        problem["num_segments"],
        limits,
        degree=8,
        terminal_order=2,
        continuity_order=4,
    )
    traj, seg_time, _, _ = upd.solve({})
    n = problem["num_segments"]
    s = traj.derivative().derivative().derivative().derivative()
    sc = 1.0 / (n * seg_time)
    assert np.max(np.abs(sample(s))) * sc**4 <= 60.0 + 1e-1


def test_symbolic_and_matrix_agree_with_snap(problem):
    limits = Limits(box(1.0, 2), box(2.0, 2), jerk=box(6.0, 2), snap=box(60.0, 2))
    kw = dict(degree=9)
    matrix = TrajectoryUpdater(
        problem["source"],
        problem["target"],
        problem["domain"],
        problem["num_segments"],
        limits,
        use_symbolic_constraints=False,
        **kw,
    )
    symbolic = TrajectoryUpdater(
        problem["source"],
        problem["target"],
        problem["domain"],
        problem["num_segments"],
        limits,
        use_symbolic_constraints=True,
        **kw,
    )
    _, st_m, cost_m, _ = matrix.solve({})
    _, st_s, cost_s, _ = symbolic.solve({})
    assert st_m == pytest.approx(st_s, rel=1e-4)
    assert cost_m == pytest.approx(cost_s, rel=1e-4)


def test_continuity_beyond_constrained_derivative_raises(problem):
    limits = Limits(problem["velocity_set"], problem["acceleration_set"])  # J = 2
    with pytest.raises(ValueError, match="continuity_order"):
        TrajectoryUpdater(
            problem["source"],
            problem["target"],
            problem["domain"],
            problem["num_segments"],
            limits,
            degree=problem["degree"],
            continuity_order=3,
        )


def test_terminal_order_default_and_configurable(problem):
    # default terminal_order = J = 2 -> zero terminal velocity AND acceleration
    upd = make_updater(problem)
    traj, seg_time, _, _ = upd.solve({})
    n = problem["num_segments"]
    v = traj.derivative()
    a = v.derivative()
    sc = 1.0 / (n * seg_time)
    assert np.linalg.norm(sample(v)[0]) * sc < 1e-4  # r'(0) = 0
    assert np.linalg.norm(sample(a)[0]) * sc**2 < 1e-3  # r''(0) = 0
    # terminal_order = 1 pins only velocity; still solves.
    upd1 = make_updater(problem, terminal_order=1)
    assert upd1.terminal_order == 1
    traj1, seg_time1, _, _ = upd1.solve({})
    v1 = traj1.derivative()
    assert np.linalg.norm(sample(v1)[0]) / (n * seg_time1) < 1e-4
