"""Integration tests for the SCS minimum-time planner.

These exercise the full region-construction + scsplanning pipeline. The oracles
are end-to-end physical: the returned trajectory must reach the goal, stay
collision-free, and respect the velocity/acceleration limits in physical time.
"""

import numpy as np
import pydrake.all as pd
import pytest

from pybmtp import EISCSPlanner, Limits, SCSResult, solve_scs_minimum_time
from pybmtp.collisions import intersect_with_hpolyhedra
from pybmtp.regions import add_segment_splitting_hyperplanes, construct_regions_from_obstacles


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


# --- region construction unit tests -----------------------------------------


def test_regions_no_obstacles_are_the_domain():
    domain = box([-5, -5], [5, 5])
    path = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    regions = construct_regions_from_obstacles(path, [], domain)
    assert len(regions) == len(path) - 1
    for r in regions:
        # Each region is exactly the domain (same number of faces).
        assert r.A().shape == domain.A().shape


def test_regions_contain_their_segment_endpoints():
    domain = box([-5, -5], [5, 5])
    obstacle = box([-0.5, -0.6], [0.5, 0.6])
    path = np.array([[-2.0, 0.0], [0.0, 1.2], [2.0, 0.0]])
    regions = construct_regions_from_obstacles(path, [obstacle], domain, margin=1e-3)
    for j, r in enumerate(regions):
        for wp in (path[j], path[j + 1]):
            assert np.all(r.A() @ wp - r.b() <= 1e-6), f"region {j} excludes a segment endpoint"


def test_regions_separate_the_obstacle():
    # Every constructed region must exclude the obstacle interior.
    domain = box([-5, -5], [5, 5])
    obstacle = box([-0.5, -0.6], [0.5, 0.6])
    path = np.array([[-2.0, 0.0], [0.0, 1.2], [2.0, 0.0]])
    regions = construct_regions_from_obstacles(path, [obstacle], domain, margin=1e-3)
    obs_center = np.array([0.0, 0.0])
    for r in regions:
        # Obstacle center must be cut off by at least one half-space.
        assert np.any(r.A() @ obs_center - r.b() > 0)


def test_splitting_makes_non_adjacent_regions_disjoint():
    domain = box([-5, -5], [5, 5])
    path = np.array([[-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    regions = construct_regions_from_obstacles(path, [], domain)
    split = add_segment_splitting_hyperplanes(path, regions, split_margin=1e-3)
    # regions[0] and regions[2] must no longer intersect.
    assert not split[0].IntersectsWith(split[2])


# --- end-to-end planner tests ------------------------------------------------


def test_obstacle_free_straight_path():
    dim = 2
    v, a = vel_acc(dim)
    domain = box([-5, -5], [5, 5])
    path = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    res = EISCSPlanner().solve(path, [], domain, Limits(v, a))

    assert isinstance(res, SCSResult)
    np.testing.assert_allclose(res.trajectory(res.trajectory.initial_time), path[0], atol=1e-4)
    np.testing.assert_allclose(res.trajectory(res.trajectory.final_time), path[-1], atol=1e-4)
    assert res.total_time > 0
    assert res.timing["total"] > 0


def test_velocity_acceleration_respected_in_physical_time():
    dim = 2
    v, a = vel_acc(dim, vmax=1.0, amax=2.0)
    domain = box([-5, -5], [5, 5])
    path = np.array([[-2.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    res = solve_scs_minimum_time(path, [], domain, Limits(v, a))

    # scsplanning returns the trajectory already in physical seconds.
    vel = res.trajectory.derivative()
    acc = vel.derivative()
    assert np.max(np.abs(sample(vel))) <= 1.0 + 1e-2
    assert np.max(np.abs(sample(acc))) <= 2.0 + 5e-2


def test_avoids_obstacle_with_routed_path():
    dim = 2
    v, a = vel_acc(dim)
    domain = box([-5, -5], [5, 5])
    obstacle = box([-0.5, -0.6], [0.5, 0.6])
    # Seed waypoints route above the box top (y = 0.6).
    path = np.array([[-2.0, 0.0], [0.0, 1.2], [2.0, 0.0]])
    res = EISCSPlanner().solve(path, [obstacle], domain, Limits(v, a))

    hits = intersect_with_hpolyhedra(res.trajectory, [obstacle], tol=1e-3)
    assert hits == {}, f"trajectory collides with obstacle: {hits}"
    assert res.num_regions >= 1


def test_polygonal_solver_type():
    dim = 2
    v, a = vel_acc(dim)
    domain = box([-5, -5], [5, 5])
    path = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    res = EISCSPlanner(solver_type="polygonal").solve(path, [], domain, Limits(v, a))
    assert res.solver_type == "polygonal"
    np.testing.assert_allclose(res.trajectory(res.trajectory.final_time), path[-1], atol=1e-4)


def test_rejects_non_hpolyhedron_domain():
    dim = 2
    v, a = vel_acc(dim)
    domain = pd.Hyperellipsoid.MakeHypersphere(5.0, np.zeros(dim))
    path = np.array([[-2.0, 0.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="HPolyhedron"):
        EISCSPlanner().solve(path, [], domain, Limits(v, a))


def test_rejects_unknown_solver_type():
    with pytest.raises(ValueError, match="solver_type"):
        EISCSPlanner(solver_type="bogus")
