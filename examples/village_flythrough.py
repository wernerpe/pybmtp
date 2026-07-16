"""Minimum-time UAV flight across the "village" — a jerk/snap-constrained demo.

The village (buildings, trees, bushes) is the benchmark environment from Fast
Path Planning; the scene lives in ``village/`` (see its LICENSE — reproduced
under Apache-2.0). This example shows *our* solution to the problem, set up
exactly as in the paper's village experiment:

  * a simple three-corner route around the village is arc-length interpolated
    into a handful of long segments (``arc_length_interpolate``);
  * pybmt's biconvex ``MinimumTimePlanner`` flies it in minimum time subject to
    velocity, acceleration, jerk **and snap** limits — the snap bound being the
    differential-flatness stand-in for a quadrotor's actuator limit — separating
    against the village obstacles as it goes.

No external planner is used: the obstacles come straight from the village scene
and pybmt does the rest. It saves a top-view + altitude figure and (unless
``--no-meshcat``) publishes a meshcat animation of the drone tracing the path.

    cd examples && uv run village_flythrough.py
    cd examples && uv run village_flythrough.py --no-meshcat
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import pydrake.all as pd
from village import Village, plot_composite_bezier_curve

from pybmt import Limits, MinimumTimePlanner
from pybmt.collisions import intersect_with_hpolyhedra

# --- village + flight configuration (matches the paper's village experiment) ---
VILLAGE = dict(village_height=10, village_side=34, building_every=7, density=0.3, seed=0)
V_MAX, A_MAX, J_MAX, SNAP_MAX = 5.0, 5.0, 25.0, 50.0
TRAJ_DEGREE = 8  # C4-capable; rest velocity+acceleration at the ends
SEGMENT_LENGTH = 10.0  # arc-length interpolation spacing -> few long segments


def sphere(radius, dim=3):
    """Isotropic limit set {x : ||x|| <= radius} as a Drake Hypersphere."""
    return pd.Hyperellipsoid.MakeHypersphere(radius, np.zeros(dim))


def village_obstacles(village):
    """The village's solid boxes (buildings, trees, bushes, ground) as HPolyhedra."""
    return [pd.HPolyhedron.MakeBox(l, u) for l, u in zip(village.L, village.U)]


def arc_length_interpolate(path, target_segment_length):
    """Subdivide each segment of ``path`` (N, dim) so no piece exceeds
    ``target_segment_length``, preserving the original corner knots. This keeps
    the segments long (well-conditioned high-derivative constraints) while still
    following the requested route."""
    path = np.asarray(path, float)
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        length = float(np.linalg.norm(b - a))
        if length > target_segment_length:
            n = int(np.ceil(length / target_segment_length))
            for t in np.linspace(0.0, 1.0, n + 1)[1:-1]:
                out.append(a * (1 - t) + b * t)
        out.append(b)
    return np.array(out)


def render_figure(path_out, village, waypoints, trajectory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, (ax_xy, ax_z) = plt.subplots(1, 2, figsize=(14, 6))

    for l, u in zip(village.L, village.U):
        if u[2] - l[2] > 1.0:  # tall solids ~ buildings / trees
            ax_xy.add_patch(
                mpatches.Rectangle(l[:2], *(u[:2] - l[:2]), facecolor="0.75", edgecolor="none")
            )
    ts = np.linspace(trajectory.initial_time, trajectory.final_time, 500)
    xyz = np.array([trajectory(t) for t in ts])
    ax_xy.plot(waypoints[:, 0], waypoints[:, 1], "o--", color="0.4", ms=3, lw=1, label="route")
    ax_xy.plot(xyz[:, 0], xyz[:, 1], "-", color="tab:blue", lw=2, label="min-time trajectory")
    ax_xy.set_aspect("equal")
    ax_xy.set_title("top view")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.legend(loc="upper left", fontsize=8)

    T = trajectory.final_time
    speed = np.array([np.linalg.norm(trajectory.derivative()(t)) for t in ts])
    ax_z.plot(ts, xyz[:, 2], color="tab:blue", label="altitude z(t)")
    ax_z.set_xlabel("time [s]")
    ax_z.set_ylabel("z", color="tab:blue")
    ax_z2 = ax_z.twinx()
    ax_z2.plot(ts, speed, color="tab:red", lw=1)
    ax_z2.axhline(V_MAX, color="tab:red", ls=":", lw=1)
    ax_z2.set_ylabel("speed", color="tab:red")
    ax_z.set_title(f"min time = {T:.2f} s")

    fig.tight_layout()
    fig.savefig(path_out, dpi=130)
    plt.close(fig)
    return path_out


def animate_meshcat(village, trajectory, fps=30):
    from meshcat.animation import Animation
    from meshcat.geometry import MeshLambertMaterial, Sphere
    from meshcat.transformations import translation_matrix

    plot_composite_bezier_curve(village, trajectory, "flight", color=[0.1, 0.3, 1.0], size=0.06)
    village["drone"].set_object(Sphere(0.25), MeshLambertMaterial(color="0x1133ff"))

    T = trajectory.final_time
    n = max(2, int(T * fps))
    anim = Animation()
    for k in range(n + 1):
        with anim.at_frame(village, k) as frame:
            frame["drone"].set_transform(translation_matrix(trajectory(T * k / n)))
    village.set_animation(anim)


def main(show_meshcat=True, out="village_flythrough.png"):
    village = Village()
    village.build(**VILLAGE)
    obstacles = village_obstacles(village)
    domain = village.domain
    side, height = VILLAGE["village_side"], VILLAGE["village_height"]

    # Three-corner route around the village, arc-length interpolated (as in the paper).
    corners = np.array(
        [
            [-0.2, -0.2, 2.0],
            [-0.2, side + 0.2, 2.0 + (height - 4) / 2],
            [side + 0.2, side + 0.2, height - 2.0],
        ]
    )
    waypoints = arc_length_interpolate(corners, SEGMENT_LENGTH)
    print(
        f"[village] {len(obstacles)} obstacles, {len(waypoints)} waypoints "
        f"({len(waypoints) - 1} segments)"
    )

    limits = Limits(sphere(V_MAX), sphere(A_MAX), jerk=sphere(J_MAX), snap=sphere(SNAP_MAX))
    t0 = time.perf_counter()
    res = MinimumTimePlanner(
        trajectory_degree=TRAJ_DEGREE,
        continuity_order=4,  # C1..C4 continuous (continuous snap)
        terminal_order=2,  # rest velocity + acceleration at both ends
        plane_degree=1,
        rel_term=0.01,
        max_iter=50,
    ).solve(waypoints, obstacles, domain, limits, verbose=True)
    solve_s = time.perf_counter() - t0

    hits = intersect_with_hpolyhedra(res.trajectory, obstacles, tol=1e-3)
    snap = res.trajectory.derivative().derivative().derivative().derivative()
    ts = np.linspace(res.trajectory.initial_time, res.trajectory.final_time, 2000)
    peak_snap = max(np.linalg.norm(snap(t)) for t in ts)
    print(
        f"[village] min time = {res.total_time:.2f} s  |  solve {solve_s:.2f} s  |  "
        f"{res.num_iterations} iters  |  peak |snap| = {peak_snap:.1f} (<= {SNAP_MAX})  |  "
        f"collision-free: {hits == {}}"
    )

    fig_path = pathlib.Path(out).resolve()
    render_figure(fig_path, village, waypoints, res.trajectory)
    print(f"[village] figure -> {fig_path}")

    if show_meshcat:
        animate_meshcat(village, res.trajectory)
        print(f"[village] meshcat animation at {village.url()}")
        input("press <enter> to exit...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-meshcat", action="store_true", help="run headless (figure only)")
    ap.add_argument("--out", default="village_flythrough.png", help="output figure path")
    args = ap.parse_args()
    main(show_meshcat=not args.no_meshcat, out=args.out)
