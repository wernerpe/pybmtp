"""Guiding-example integration test (2D convex-obstacle scene).

The scene (5 box obstacles in a 10x10 domain) and the 7-waypoint seed path are
lifted from the original ``mmt_gcs`` ``debug_guiding_example.py``. The seed path
is collision-free and routes around the obstacles; the BMTP planner refines it
into a minimum-time trajectory.

The test renders the obstacles, the seed path, and the optimized trajectory to a
PNG under ``<repo>/test_outputs/`` (git-ignored) so the result can be inspected
by eye. Set ``PYBMTP_TEST_OUTPUT_DIR`` to override the location.
"""

import os
import pathlib

import numpy as np
import pydrake.all as pd
import pytest

from pybmtp import Limits, MinimumTimePlanner
from pybmtp.collisions import intersect_with_hpolyhedra


def _output_dir() -> pathlib.Path:
    override = os.environ.get("PYBMTP_TEST_OUTPUT_DIR")
    out = (
        pathlib.Path(override)
        if override
        else pathlib.Path(__file__).resolve().parents[2] / "test_outputs"
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


def _box(center, w, h):
    """Axis-aligned box HPolyhedron from center + width/height (returns lb, ub, set)."""
    c = np.asarray(center, float)
    lb = c - np.array([w / 2, h / 2])
    ub = c + np.array([w / 2, h / 2])
    return lb, ub, pd.HPolyhedron.MakeBox(lb, ub)


def guiding_example_scene():
    """Return ``(domain, obstacles, obstacle_boxes, waypoints, vel_set, acc_set)``.

    ``obstacle_boxes`` is the list of ``(lb, ub)`` pairs kept alongside the
    HPolyhedra purely so the obstacles can be drawn as rectangles.
    """
    domain = pd.HPolyhedron.MakeBox([0.0, 0.0], [10.0, 10.0])

    # Obstacle (center, width, height) exactly as in guiding_example.ipynb.
    specs = [
        ((5.25, 2.0), 7.5, 1.0),
        ((3.0, 5.0), 5.0, 1.0),
        ((7.6, 4.1), 0.8, 1.5),
        ((8.5, 6.0), 1.2, 1.2),
        ((7.0, 8.0), 4.8, 0.8),
    ]
    obstacle_boxes, obstacles = [], []
    for center, w, h in specs:
        lb, ub, hpoly = _box(center, w, h)
        obstacle_boxes.append((lb, ub))
        obstacles.append(hpoly)

    waypoints = np.array(
        [
            [1.0, 1.0],
            [1.0, 3.0],
            [9.5, 3.0],
            [9.5, 7.0],
            [3.0, 7.0],
            [3.0, 9.0],
            [9.0, 9.0],
        ]
    )

    # Derivative constraints exactly as in guiding_example.ipynb:
    # isotropic velocity ball of radius v_max=10, acceleration ball a_max=30.
    vel_set = pd.Hyperellipsoid.MakeHypersphere(10.0, np.zeros(2))
    acc_set = pd.Hyperellipsoid.MakeHypersphere(30.0, np.zeros(2))
    return domain, obstacles, obstacle_boxes, waypoints, vel_set, acc_set


def _render(path_out, obstacle_boxes, waypoints, seed_curve, opt_curve, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    for lb, ub in obstacle_boxes:
        ax.add_patch(
            mpatches.Rectangle(
                lb,
                ub[0] - lb[0],
                ub[1] - lb[1],
                facecolor="lightcoral",
                edgecolor="k",
                linewidth=1.5,
                zorder=5,
            )
        )

    seed = np.array(
        [seed_curve(t) for t in np.linspace(seed_curve.initial_time, seed_curve.final_time, 200)]
    )
    ax.plot(
        seed[:, 0],
        seed[:, 1],
        color="k",
        linewidth=1.5,
        linestyle="--",
        label="seed path",
        zorder=8,
    )
    ax.scatter(waypoints[:, 0], waypoints[:, 1], color="k", s=40, zorder=9)

    opt = np.array(
        [opt_curve(t) for t in np.linspace(opt_curve.initial_time, opt_curve.final_time, 400)]
    )
    ax.plot(opt[:, 0], opt[:, 1], color="red", linewidth=2.5, label="BMTP trajectory", zorder=10)
    ax.scatter([waypoints[0, 0]], [waypoints[0, 1]], color="k", s=160, zorder=11)
    ax.scatter(
        [waypoints[-1, 0]],
        [waypoints[-1, 1]],
        marker="*",
        color="gold",
        edgecolor="k",
        s=420,
        zorder=11,
    )

    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path_out, dpi=130)
    plt.close(fig)


def test_guiding_example_bcp_solves_and_renders():
    pytest.importorskip("matplotlib")
    domain, obstacles, obstacle_boxes, waypoints, vel_set, acc_set = guiding_example_scene()

    # collision_check_tol must be at least as fine as the final assertion tol,
    # else an aggressive (a_max=30) corner-cut can graze an obstacle below the
    # planner's subdivision resolution. A small plane buffer keeps margin.
    # rel_term=0.01 converges past the early-termination band where the cost is
    # sensitive to parallel-solve nondeterminism, giving a reproducible optimum.
    res = MinimumTimePlanner(
        rel_term=0.01,
        max_iter=120,
        collision_check_tol=1e-3,
        trajectory_to_plane_tol=1e-3,
    ).solve(waypoints, obstacles, domain, Limits(vel_set, acc_set))

    # Final trajectory must be collision-free and hit start/goal.
    hits = intersect_with_hpolyhedra(res.feasible_trajectories[-1], obstacles, tol=1e-3)
    assert hits == {}, f"trajectory collides: {hits}"
    np.testing.assert_allclose(res.trajectory(res.trajectory.initial_time), waypoints[0], atol=1e-4)
    np.testing.assert_allclose(res.trajectory(res.trajectory.final_time), waypoints[-1], atol=1e-4)

    # Cost regression: the converged minimum time is ~2.45s for this scene
    # (v_max=10, a_max=30). Guard against silent regressions in the solve.
    assert (
        2.40 < res.total_time < 2.52
    ), f"unexpected minimum time {res.total_time:.3f}s (expected ~2.45s)"

    out = _output_dir() / "guiding_example_bcp.png"
    _render(
        out,
        obstacle_boxes,
        waypoints,
        res.feasible_trajectories[0],
        res.trajectory,
        title=f"BMTP guiding example  (T={res.total_time:.2f}s, " f"{res.num_iterations} iters)",
    )
    assert out.exists() and out.stat().st_size > 0
    print(f"[guiding-example] wrote {out}")
