"""Integration tests for the biconvex MinimumTimePlanner.

These exercise the full outer loop. The oracles are end-to-end physical:
the returned trajectory must reach the goal, stay collision-free, respect the
velocity/acceleration limits in physical time, and (when an obstacle blocks the
straight path) actually detour around it.
"""

import numpy as np
import pydrake.all as pd

from pybmt import MinimumTimePlanner, SolveResult, solve_minimum_time
from pybmt.collisions import intersect_with_hpolyhedra


def box(lb, ub):
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    d = lb.size
    A = np.vstack([np.eye(d), -np.eye(d)])
    return pd.HPolyhedron(A, np.concatenate([ub, -lb]))


def vel_acc(dim, vmax=1.0, amax=2.0):
    return box([-vmax] * dim, [vmax] * dim), box([-amax] * dim, [amax] * dim)


def sample(curve, n=600):
    ts = np.linspace(curve.initial_time, curve.final_time, n)
    return np.array([curve(t) for t in ts])


def test_obstacle_free_straight_path():
    dim = 2
    v, a = vel_acc(dim)
    domain = box([-5, -5], [5, 5])
    path = np.array([[-2.0, 0.0], [2.0, 0.0]])
    res = MinimumTimePlanner(rel_term=0.05, max_iter=50).solve(path, [], domain, v, a)

    assert isinstance(res, SolveResult)
    np.testing.assert_allclose(res.trajectory(res.trajectory.initial_time), path[0], atol=1e-5)
    np.testing.assert_allclose(res.trajectory(res.trajectory.final_time), path[-1], atol=1e-5)
    assert res.total_time > 0
    assert res.timing["total"] > 0


def test_velocity_acceleration_respected_in_physical_time():
    dim = 2
    v, a = vel_acc(dim, vmax=1.0, amax=2.0)
    domain = box([-5, -5], [5, 5])
    path = np.array([[-2.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    res = solve_minimum_time(path, [], domain, v, a)

    # trajectory is already in physical seconds.
    vel = res.trajectory.derivative()
    acc = vel.derivative()
    assert np.max(np.abs(sample(vel))) <= 1.0 + 1e-2
    assert np.max(np.abs(sample(acc))) <= 2.0 + 5e-2


def test_detours_around_blocking_obstacle():
    dim = 2
    v, a = vel_acc(dim)
    domain = box([-5, -5], [5, 5])
    # Box squarely on the straight line from start to goal. BCP refines a
    # collision-free polygonal seed; it does not synthesize a detour from a
    # colliding straight line. So the seed waypoints route above the box top
    # (y = 0.6), and the planner tightens that into a minimum-time curve while
    # keeping it on the safe side of the separating planes.
    obstacle = box([-0.5, -0.6], [0.5, 0.6])
    path = np.array([[-2.0, 0.0], [0.0, 1.2], [2.0, 0.0]])
    res = MinimumTimePlanner(rel_term=0.05, max_iter=80).solve(path, [obstacle], domain, v, a)

    # Final trajectory must be collision-free against the obstacle.
    hits = intersect_with_hpolyhedra(res.feasible_trajectories[-1], [obstacle], tol=1e-3)
    assert hits == {}, f"final trajectory still collides: {hits}"
    # And it must have separating planes recorded.
    assert len(res.planes) >= 1


def test_converges_and_is_monotone_in_feasible_time():
    dim = 2
    v, a = vel_acc(dim)
    domain = box([-5, -5], [5, 5])
    obstacle = box([-0.5, -0.6], [0.5, 0.6])
    path = np.array([[-2.0, 0.0], [0.0, 1.2], [2.0, 0.0]])
    res = MinimumTimePlanner(rel_term=0.05, max_iter=80).solve(path, [obstacle], domain, v, a)

    assert res.converged
    # Feasible segment times should be non-increasing (planner only accepts
    # faster feasible iterates).
    times = res.feasible_segment_times
    assert all(t2 <= t1 + 1e-9 for t1, t2 in zip(times, times[1:]))
