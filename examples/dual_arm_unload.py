# /// script
# requires-python = ">=3.10"
# dependencies = ["pybmt"]
#
# [tool.uv.sources]
# pybmt = { path = "..", editable = true }
# ///
"""Dual-arm pallet unload mirrored as a pure-geometry BCP problem.

Two arms (red, blue) each lift one brick off a shared pallet and carry it to a
color-matched offload zone. There is no diff-IK and no URDF here: the planner
works directly on the 6D task-space state ``[red_xyz, blue_xyz]`` (each arm's
*brick center*), and every collision pair is a configuration-space HPolyhedron
built from box Minkowski sums.

The script samples a random unloadable pallet, plans the simultaneous two-arm
move, verifies the result is collision free, and publishes a meshcat animation
of the bricks and grippers.

Choose the planner with ``--planner``: ``bcp`` (default, the biconvex
MinimumTimePlanner) or ``scs`` (the SCSPlanner baseline, which adds
Assumption-1 splitting planes between non-adjacent convex regions).

    uv run examples/dual_arm_unload.py
    uv run examples/dual_arm_unload.py --planner scs --seed 3
    # or, with pybmt already installed:
    python examples/dual_arm_unload.py
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import pydrake.all as pd

from pybmt import MinimumTimePlanner, SCSPlanner
from pybmt.collisions import intersect_with_hpolyhedra

# --- scene constants (mirrors the hardware sim) ---------------------------
PALLET_POS = np.array([0.55, 0.2, 0.01])      # center; 2cm board on the floor
PALLET_DIM = np.array([0.4, 0.6, 0.02])
ZONE_DIM = np.array([0.5, 0.5, 0.02])
LEFT_ZONE = np.array([0.1, -0.45])            # blue drop
RIGHT_ZONE = np.array([1.0, -0.45])           # red drop
GRIPPER_DIM = np.array([0.15, 0.15, 0.30])
Z_UB = 0.75                                   # domain ceiling


# --- pure-geometry configuration-space obstacle builders ------------------
# Each builder returns an HPolyhedron in the 6D state [p_red; p_blue] (or its
# 3D marginal embedded into 6D) whose interior is the set of *colliding*
# states; the planner separates the trajectory from it.

def _c_obstacle_from_box_pair(dims1, dims2):
    """6D HPolyhedron for two moving boxes: A(p2 - p1) <= b."""
    half = np.asarray(dims1) / 2 + np.asarray(dims2) / 2
    hbox = pd.Hyperrectangle(-half, half).MakeHPolyhedron()
    A, b = hbox.A(), hbox.b()
    return pd.HPolyhedron(np.concatenate((-A, A), axis=1), b)


def _fixed_box_obstacle(moving_dims, fixed_dims, fixed_center, robot_idx,
                        moving_offset=np.zeros(3)):
    """6D HPolyhedron for a moving box (optionally offset, e.g. a gripper) vs a
    fixed box. Marginalizes out the fixed center and embeds into the 6D slot of
    the given arm."""
    c6 = _c_obstacle_from_box_pair(moving_dims, fixed_dims)
    A, b = c6.A(), c6.b()
    A_move, A_fixed = A[:, :3], A[:, 3:]
    b_new = b - A_move @ moving_offset - A_fixed @ np.asarray(fixed_center)
    z = np.zeros((A_move.shape[0], 3))
    A6 = np.hstack([A_move, z]) if robot_idx == 0 else np.hstack([z, A_move])
    return pd.HPolyhedron(A6, b_new)


def _box_to_box_obstacle(dims0, dims1, offset0=np.zeros(3), offset1=np.zeros(3)):
    """6D HPolyhedron for the two moving boxes (each optionally offset)."""
    c6 = _c_obstacle_from_box_pair(dims0, dims1)
    A, b = c6.A(), c6.b()
    b_new = b - A @ np.concatenate([offset0, offset1])
    return pd.HPolyhedron(A, b_new)


def build_obstacles(red_dim, blue_dim, stationary, gripper_offsets):
    """All collision pairs for the simultaneous two-arm move."""
    obstacles = []
    moving = [(0, red_dim), (1, blue_dim)]
    # moving brick / gripper vs each stationary brick on the pallet
    for idx, dim in moving:
        off = gripper_offsets[idx]
        for s_dim, s_center in stationary:
            obstacles.append(_fixed_box_obstacle(dim, s_dim, s_center, idx))
            obstacles.append(
                _fixed_box_obstacle(GRIPPER_DIM, s_dim, s_center, idx, off))
    # moving brick vs moving brick, and gripper vs gripper
    obstacles.append(_box_to_box_obstacle(red_dim, blue_dim))
    obstacles.append(_box_to_box_obstacle(
        GRIPPER_DIM, GRIPPER_DIM, gripper_offsets[0], gripper_offsets[1]))
    return obstacles


# --- random unloadable pallet --------------------------------------------

def sample_pallet(rng, n_bricks=5):
    """Place ``n_bricks`` non-overlapping bricks on a jittered grid covering the
    pallet, color them, and choose one red + one blue to unload. Returns the two
    moving bricks and the list of stationary ``(dim, center)`` pairs."""
    half = PALLET_DIM[:2] / 2
    cols = np.linspace(-half[0] * 0.7, half[0] * 0.7, 3)
    rows = np.linspace(-half[1] * 0.8, half[1] * 0.8, 3)
    cells = [(cx, ry) for cx in cols for ry in rows]
    rng.shuffle(cells)

    pallet_top = PALLET_POS[2] + PALLET_DIM[2] / 2
    bricks = []
    for cx, ry in cells[:n_bricks]:
        dim = rng.uniform([0.05, 0.05, 0.08], [0.07, 0.07, 0.12])
        jit = rng.uniform(-0.02, 0.02, size=2)
        center = np.array([PALLET_POS[0] + cx + jit[0],
                           PALLET_POS[1] + ry + jit[1],
                           pallet_top + dim[2] / 2])
        bricks.append({"dim": dim, "center": center})

    colors = ["red", "blue"] + [rng.choice(["red", "blue"])
                                for _ in range(n_bricks - 2)]
    rng.shuffle(colors)
    for b, c in zip(bricks, colors):
        b["color"] = c

    red = next(b for b in bricks if b["color"] == "red")
    blue = next(b for b in bricks if b["color"] == "blue")
    stationary = [(b["dim"], b["center"]) for b in bricks
                  if b is not red and b is not blue]
    return red, blue, stationary, bricks


def drop_center(zone_xy, brick_dim):
    """Brick-center goal resting in an offload zone."""
    zone_top = 0.0 + ZONE_DIM[2]  # zone base sits on the floor (z=0)
    return np.array([zone_xy[0], zone_xy[1], zone_top + brick_dim[2] / 2])


# --- forward unload waypoints (center-based) ------------------------------

def _single_arm_waypoints(start, goal, lift_h):
    """grasp -> lift -> move toward bins -> hover over goal -> lower."""
    intermediate_y = -0.325
    return np.array([
        start,
        [start[0], start[1], lift_h],
        [start[0], intermediate_y, lift_h],
        [goal[0], goal[1], lift_h],
        goal,
    ])


def make_waypoints(red_start, red_goal, blue_start, blue_goal, lift_h):
    """Sequential coordination: the arm with smaller Y (closer to the bins)
    moves first while the other waits, then they swap roles."""
    red_wps = _single_arm_waypoints(red_start, red_goal, lift_h)
    blue_wps = _single_arm_waypoints(blue_start, blue_goal, lift_h)
    red_first = red_start[1] < blue_start[1]

    if red_first:
        first, second = red_wps, blue_wps
        first_goal, second_start = red_goal, blue_start
    else:
        first, second = blue_wps, red_wps
        first_goal, second_start = blue_goal, red_start

    rows = []
    for wp in first:                       # first arm moves, second waits
        rows.append(np.concatenate([wp, second_start] if red_first
                                   else [second_start, wp]))
    for wp in second[1:]:                  # second arm moves, first holds goal
        rows.append(np.concatenate([first_goal, wp] if red_first
                                   else [wp, first_goal]))
    return np.array(rows)


# --- seed vs. optimized plot ----------------------------------------------

def plot_trajectory(path_out, waypoints, trajectory, planner):
    """3D plot of each arm's brick-center path: dashed waypoint seed vs. the
    solid optimized trajectory (red arm in dims 0:3, blue arm in dims 3:6)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    ts = np.linspace(trajectory.initial_time, trajectory.final_time, 400)
    traj = np.array([np.asarray(trajectory(t)).reshape(-1) for t in ts])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for slot, name, col in [(slice(0, 3), "red", "tab:red"),
                            (slice(3, 6), "blue", "tab:blue")]:
        wp = waypoints[:, slot]
        ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], color=col, linestyle="--",
                marker="o", markersize=4, alpha=0.6,
                label=f"{name} waypoints")
        tr = traj[:, slot]
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=col, linewidth=2.5,
                label=f"{name} optimized")
        ax.scatter(*wp[0], color=col, s=80, marker="o")
        ax.scatter(*wp[-1], color=col, s=160, marker="*", edgecolor="k")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Dual-arm unload ({planner.upper()}): "
                 f"waypoints vs. optimized  "
                 f"(T={trajectory.final_time - trajectory.initial_time:.2f}s)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path_out, dpi=130)
    plt.close(fig)


# --- meshcat animation ----------------------------------------------------

def animate(meshcat, trajectory, red, blue, stationary_bricks, gripper_offsets,
            fps=60):
    def box(name, dim, color):
        meshcat.SetObject(name, pd.Box(*dim), rgba=color)

    floor = pd.Rgba(0.6, 0.5, 0.35, 1.0)
    box("scene/pallet", PALLET_DIM, floor)
    meshcat.SetTransform("scene/pallet", pd.RigidTransform(PALLET_POS))
    for tag, xy, col in [("left", LEFT_ZONE, pd.Rgba(0.2, 0.2, 0.8, 0.4)),
                         ("right", RIGHT_ZONE, pd.Rgba(0.8, 0.2, 0.2, 0.4))]:
        box(f"scene/{tag}_zone", ZONE_DIM, col)
        meshcat.SetTransform(f"scene/{tag}_zone",
                             pd.RigidTransform([xy[0], xy[1], ZONE_DIM[2] / 2]))
    for i, b in enumerate(stationary_bricks):
        col = pd.Rgba(0.8, 0.2, 0.2, 1.0) if b["color"] == "red" \
            else pd.Rgba(0.2, 0.2, 0.8, 1.0)
        box(f"scene/static_{i}", b["dim"], col)
        meshcat.SetTransform(f"scene/static_{i}", pd.RigidTransform(b["center"]))

    box("scene/red_brick", red["dim"], pd.Rgba(0.85, 0.15, 0.15, 1.0))
    box("scene/blue_brick", blue["dim"], pd.Rgba(0.15, 0.15, 0.85, 1.0))
    box("scene/red_grip", GRIPPER_DIM, pd.Rgba(0.4, 0.4, 0.4, 0.5))
    box("scene/blue_grip", GRIPPER_DIM, pd.Rgba(0.4, 0.4, 0.4, 0.5))

    t0, t1 = trajectory.initial_time, trajectory.final_time
    meshcat.StartRecording()
    for t in np.linspace(t0, t1, max(2, int((t1 - t0) * fps))):
        s = np.asarray(trajectory(t)).reshape(-1)
        red_c, blue_c = s[:3], s[3:]
        meshcat.SetTransform("scene/red_brick",
                             pd.RigidTransform(red_c), time_in_recording=t)
        meshcat.SetTransform("scene/blue_brick",
                             pd.RigidTransform(blue_c), time_in_recording=t)
        meshcat.SetTransform("scene/red_grip",
                             pd.RigidTransform(red_c + gripper_offsets[0]),
                             time_in_recording=t)
        meshcat.SetTransform("scene/blue_grip",
                             pd.RigidTransform(blue_c + gripper_offsets[1]),
                             time_in_recording=t)
    meshcat.StopRecording()
    meshcat.PublishRecording()


def main(seed=0, planner="bcp"):
    rng = np.random.default_rng(seed)
    red, blue, stationary, all_bricks = sample_pallet(rng)

    # Gripper rides above the brick center: half the brick + half the gripper.
    gripper_offsets = [
        np.array([0.0, 0.0, red["dim"][2] / 2 + GRIPPER_DIM[2] / 2]),
        np.array([0.0, 0.0, blue["dim"][2] / 2 + GRIPPER_DIM[2] / 2]),
    ]

    red_start, blue_start = red["center"], blue["center"]
    red_goal = drop_center(RIGHT_ZONE, red["dim"])
    blue_goal = drop_center(LEFT_ZONE, blue["dim"])

    lift_h = min(Z_UB - GRIPPER_DIM[2] / 2,
                 max(b["center"][2] for b in all_bricks) + 0.25)
    waypoints = make_waypoints(red_start, red_goal, blue_start, blue_goal, lift_h)

    obstacles = build_obstacles(red["dim"], blue["dim"], stationary,
                                gripper_offsets)

    z_lo = PALLET_POS[2] + PALLET_DIM[2] / 2  # bricks rest on the pallet/floor
    domain = pd.HPolyhedron.MakeBox(
        [-0.2, -0.7, z_lo - 0.05, -0.2, -0.7, z_lo - 0.05],
        [1.3, 0.6, Z_UB, 1.3, 0.6, Z_UB])
    vel_set = pd.Hyperellipsoid.MakeHypersphere(2.0, np.zeros(6))
    acc_set = pd.Hyperellipsoid.MakeHypersphere(8.0, np.zeros(6))

    print(f"unloadable pallet: {len(all_bricks)} bricks "
          f"({sum(b['color'] == 'red' for b in all_bricks)} red, "
          f"{sum(b['color'] == 'blue' for b in all_bricks)} blue), "
          f"{len(obstacles)} obstacles")

    if planner == "bcp":
        res = MinimumTimePlanner(
            rel_term=0.01, max_iter=200,
            collision_check_tol=1e-3, trajectory_to_plane_tol=1e-3,
        ).solve(waypoints, obstacles, domain, vel_set, acc_set, verbose=True)
        print(f"minimum time : {res.total_time:.3f}s over "
              f"{res.num_iterations} iters")
    else:  # scs
        res = SCSPlanner(rel_term=0.01).solve(
            waypoints, obstacles, domain, vel_set, acc_set, verbose=True)
        print(f"minimum time : {res.total_time:.3f}s over "
              f"{res.num_regions} regions")

    hits = intersect_with_hpolyhedra(res.trajectory, obstacles, tol=1e-3)
    print(f"collision-free: {hits == {}}")

    out_dir = pathlib.Path(__file__).resolve().parent / "_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"dual_arm_unload_{planner}.png"
    plot_trajectory(out, waypoints, res.trajectory, planner)
    print(f"wrote {out}")

    stationary_bricks = [b for b in all_bricks if b is not red and b is not blue]
    meshcat = pd.StartMeshcat()
    animate(meshcat, res.trajectory, red, blue, stationary_bricks,
            gripper_offsets)
    print(f"meshcat: {meshcat.web_url()}  (Ctrl-C to exit)")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", choices=["bcp", "scs"], default="bcp",
                        help="planner backend (default: bcp)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the random pallet (default: 0)")
    args = parser.parse_args()
    main(seed=args.seed, planner=args.planner)
