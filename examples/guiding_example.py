# /// script
# requires-python = ">=3.10"
# dependencies = ["pybmt", "matplotlib"]
#
# [tool.uv.sources]
# pybmt = { path = "..", editable = true }
# ///
"""Guiding example: minimum-time Bezier planning in a 2D convex-obstacle scene.

Five box obstacles sit in a 10x10 workspace; a collision-free 7-waypoint seed
path routes around them. A minimum-time planner refines that seed into a
minimum-time trajectory (velocity ball v_max=10, acceleration ball a_max=30)
that stays clear of the obstacles.

Two planners are available via ``--planner``:

  * ``bcp`` (default) — the biconvex ``MinimumTimePlanner``.
  * ``scs`` — the ``SCSPlanner`` baseline (convex regions + scsplanning, with
    Assumption-1 splitting planes added between non-adjacent regions).

Run it (renders to ``examples/_out/guiding_example_<planner>.png``):

    uv run examples/guiding_example.py            # BCP
    uv run examples/guiding_example.py --planner scs
    # or, in an environment where pybmt is already installed:
    python examples/guiding_example.py
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pydrake.all as pd

from pybmt import MinimumTimePlanner, PolygonalInitializer, SCSPlanner
from pybmt.collisions import intersect_with_hpolyhedra


def build_scene():
    """Return ``(domain, obstacles, obstacle_boxes, waypoints, vel_set, acc_set)``."""
    domain = pd.HPolyhedron.MakeBox([0.0, 0.0], [10.0, 10.0])

    # (center, width, height) for each obstacle box.
    specs = [
        ((5.25, 2.0), 7.5, 1.0),
        ((3.0, 5.0), 5.0, 1.0),
        ((7.6, 4.1), 0.8, 1.5),
        ((8.5, 6.0), 1.2, 1.2),
        ((7.0, 8.0), 4.8, 0.8),
    ]
    obstacle_boxes, obstacles = [], []
    for center, w, h in specs:
        c = np.asarray(center, float)
        lb = c - np.array([w / 2, h / 2])
        ub = c + np.array([w / 2, h / 2])
        obstacle_boxes.append((lb, ub))
        obstacles.append(pd.HPolyhedron.MakeBox(lb, ub))

    waypoints = np.array([
        [1.0, 1.0], [1.0, 3.0], [9.5, 3.0], [9.5, 7.0],
        [3.0, 7.0], [3.0, 9.0], [9.0, 9.0],
    ])

    vel_set = pd.Hyperellipsoid.MakeHypersphere(10.0, np.zeros(2))
    acc_set = pd.Hyperellipsoid.MakeHypersphere(30.0, np.zeros(2))
    return domain, obstacles, obstacle_boxes, waypoints, vel_set, acc_set


def render(path_out, obstacle_boxes, waypoints, seed_curve, opt_curve, title,
           traj_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    for lb, ub in obstacle_boxes:
        ax.add_patch(mpatches.Rectangle(
            lb, ub[0] - lb[0], ub[1] - lb[1],
            facecolor="lightcoral", edgecolor="k", linewidth=1.5, zorder=5))

    seed = np.array([seed_curve(t) for t in np.linspace(
        seed_curve.initial_time, seed_curve.final_time, 200)])
    ax.plot(seed[:, 0], seed[:, 1], color="k", linewidth=1.5,
            linestyle="--", label="seed path", zorder=8)
    ax.scatter(waypoints[:, 0], waypoints[:, 1], color="k", s=40, zorder=9)

    opt = np.array([opt_curve(t) for t in np.linspace(
        opt_curve.initial_time, opt_curve.final_time, 400)])
    ax.plot(opt[:, 0], opt[:, 1], color="red", linewidth=2.5,
            label=traj_label, zorder=10)
    ax.scatter([waypoints[0, 0]], [waypoints[0, 1]], color="k", s=160, zorder=11)
    ax.scatter([waypoints[-1, 0]], [waypoints[-1, 1]], marker="*",
               color="gold", edgecolor="k", s=420, zorder=11)

    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path_out, dpi=130)
    plt.close(fig)


def main(planner="bcp"):
    domain, obstacles, obstacle_boxes, waypoints, vel_set, acc_set = build_scene()

    # A constraint-feasible polygonal warm start, drawn as the dashed seed path.
    seed_curve = PolygonalInitializer(
        vel_set, acc_set, force_same_segment_time=True).initialize(waypoints)

    if planner == "bcp":
        res = MinimumTimePlanner(
            rel_term=0.01, max_iter=120,
            collision_check_tol=1e-3, trajectory_to_plane_tol=1e-3,
        ).solve(waypoints, obstacles, domain, vel_set, acc_set, verbose=True)
        label = "BCP"
        title = (f"BCP guiding example  (T={res.total_time:.2f}s, "
                 f"{res.num_iterations} iters)")
        print(f"minimum time : {res.total_time:.3f}s over "
              f"{res.num_iterations} iters")
    else:  # scs
        res = SCSPlanner(rel_term=0.01).solve(
            waypoints, obstacles, domain, vel_set, acc_set, verbose=True)
        label = "SCS"
        title = (f"SCS guiding example  (T={res.total_time:.2f}s, "
                 f"{res.num_regions} regions)")
        print(f"minimum time : {res.total_time:.3f}s over "
              f"{res.num_regions} regions")

    hits = intersect_with_hpolyhedra(res.trajectory, obstacles, tol=1e-3)
    print(f"collision-free: {hits == {}}")

    out_dir = pathlib.Path(__file__).resolve().parent / "_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"guiding_example_{planner}.png"
    render(out, obstacle_boxes, waypoints, seed_curve, res.trajectory,
           title=title, traj_label=f"{label} trajectory")
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", choices=["bcp", "scs"], default="bcp",
                        help="planner backend (default: bcp)")
    args = parser.parse_args()
    main(planner=args.planner)
