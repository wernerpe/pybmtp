"""Dual-arm pallet unload as a pure-geometry minimum-time problem.

Two Franka arms (red, blue) cooperatively clear four bricks off a shared pallet
and stack them into two color-matched offload bins. The whole job is five
task-space phases:

  A  home      -> grasp move-1 bricks            (empty grippers)
  B  grasp     -> offload move-1 bricks          (a brick on each arm)
  C  offload   -> grasp move-2 bricks            (empty grippers)
  D  grasp     -> offload move-2 bricks          (a brick on each arm)
  E  offload   -> home                           (empty grippers)

The planner works directly on the 6D task-space state ``[red_xyz, blue_xyz]``,
where each xyz is that arm's TCP / grasp point. Every collision pair (gripper
parts, robot bases, bin walls, pallet bricks, and the carried bricks) is a
configuration-space HPolyhedron built from box Minkowski sums. The FinRay
gripper is three axis-aligned boxes per arm; their world dimensions and offsets
from the grasp point are constants captured at the reference configuration.

Each phase is planned independently with pybmtp: ``bmtp`` (the biconvex
``MinimumTimePlanner``, default) or ``scs`` (the ``EISCSPlanner`` baseline). The
script reports per-phase and total minimum time, verifies each phase is
collision free, renders a top-view + z-profile figure, and publishes a meshcat
animation of the collision geometry (pass ``--no-meshcat`` to run headless).

    cd examples && uv run dual_arm_unload.py
    cd examples && uv run dual_arm_unload.py --planner scs
    cd examples && uv run dual_arm_unload.py --no-meshcat
    # or, with pybmtp already installed:
    python examples/dual_arm_unload.py
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import pydrake.all as pd
from eiscs import EISCSPlanner

from pybmtp import Limits, MinimumTimePlanner
from pybmtp.collisions import intersect_with_hpolyhedra

# ===========================================================================
# Scene constants
# ===========================================================================
TCP_OFFSET = np.array([0.0, 0.0, 0.125])
BRICK_DIMS = np.array([0.07, 0.10, 0.21])
BRICK_CENTROID_Z = BRICK_DIMS[2] / 2  # 0.105

PALLET_CENTER = np.array([0.55, -0.175])
PALLET_DIMS = np.array([0.58, 0.40, 0.02])
GRID_HALF_X = 0.125
GRID_HALF_Y = 0.125
BRICK_OFFSETS = np.array(
    [
        [-GRID_HALF_X, -GRID_HALF_Y],  # 0
        [+GRID_HALF_X, -GRID_HALF_Y],  # 1
        [-GRID_HALF_X, +GRID_HALF_Y],  # 2
        [+GRID_HALF_X, +GRID_HALF_Y],  # 3
    ]
)
BRICK_CENTROIDS = np.array(
    [[PALLET_CENTER[0] + dx, PALLET_CENTER[1] + dy, BRICK_CENTROID_Z] for dx, dy in BRICK_OFFSETS]
)

# TCP sits GRASP_DEPTH into the brick from its top, so the tracked grasp z is
# below the brick top by (GRASP_DEPTH - FINGER_BELOW_TCP).
TRAJ_GRASP_Z = 0.17
HELD_BRICK_TCP_OFFSET_Z = BRICK_CENTROID_Z - TRAJ_GRASP_Z  # -0.065
FINGER_BELOW_TCP = 0.06

BRICK_GRASP_POINTS = np.array(
    [[PALLET_CENTER[0] + dx, PALLET_CENTER[1] + dy, TRAJ_GRASP_Z] for dx, dy in BRICK_OFFSETS]
)

OFFLOAD_GREY = np.array([-0.05, 0.45, TRAJ_GRASP_Z])
OFFLOAD_RED = np.array([1.15, 0.45, TRAJ_GRASP_Z])
OFFLOAD_DIMS = np.array([0.30, 0.30, 0.02])
OFFLOAD_WALL_HEIGHT = 0.22
SLOT_OFFSET_X = 0.07  # outer (move-1) vs inner (move-2) slot per pad

# Ferry-lane formula uses this single conservative gripper AABB.
GRIPPER_DIMS_PLANNER = np.array([0.15, 0.15, 0.30])

V_MAX = 0.40
A_MAX = 0.8

# Layout GRGR: pallet positions [FL, FR, BL, BR] = grey, red, grey, red.
# Move 1 grabs bricks (0, 1); move 2 grabs (2, 3). Within each move, the blue
# arm takes the lower id (left column) and red takes the higher id.
M1_BLUE, M1_RED = 0, 1
M2_BLUE, M2_RED = 2, 3
BRICK_COLOR = {0: "grey", 1: "red", 2: "grey", 3: "red"}

# ---------------------------------------------------------------------------
# Geometry captured at the reference configuration (FinRay parts in world frame,
# shifted robot-base boxes, and the home/hover grasp positions).
# ---------------------------------------------------------------------------
ARM_RED_PARTS = [
    ("hand", np.array([0.240080, 0.200096, 0.280000]), np.array([0.000000, -0.000000, 0.200000])),
    (
        "finger_L",
        np.array([0.040012, 0.030016, 0.120000]),
        np.array([-0.060000, -0.000024, -0.019000]),
    ),
    (
        "finger_R",
        np.array([0.040012, 0.030016, 0.120000]),
        np.array([0.060000, 0.000024, -0.019000]),
    ),
]
ARM_BLUE_PARTS = [
    ("hand", np.array([0.240080, 0.200096, 0.280000]), np.array([0.000000, 0.000000, 0.200000])),
    (
        "finger_L",
        np.array([0.040012, 0.030016, 0.120000]),
        np.array([0.060000, 0.000024, -0.019000]),
    ),
    (
        "finger_R",
        np.array([0.040012, 0.030016, 0.120000]),
        np.array([-0.060000, -0.000024, -0.019000]),
    ),
]

BASE_RED = (np.array([0.600000, 0.390000, 0.900000]), np.array([1.300000, 0.000000, 0.450000]))
BASE_BLUE = (np.array([0.600000, 0.390000, 0.900000]), np.array([-0.200000, 0.000000, 0.450000]))

P_RED_QREF = np.array([0.752977, 0.000000, 0.465270])
P_BLUE_QREF = np.array([0.347023, -0.000000, 0.465270])

BASES = [("base_blue", BASE_BLUE[0], BASE_BLUE[1]), ("base_red", BASE_RED[0], BASE_RED[1])]


# ===========================================================================
# Configuration-space obstacle primitives (box Minkowski sums)
# ===========================================================================
def c_obstacle_from_box_pair(dims1, dims2):
    """6D HPolyhedron of colliding states for two moving boxes: A(p2 - p1) <= b."""
    half = np.asarray(dims1) / 2 + np.asarray(dims2) / 2
    hbox = pd.Hyperrectangle(-half, half).MakeHPolyhedron()
    A, b = hbox.A(), hbox.b()
    return pd.HPolyhedron(np.concatenate((-A, A), axis=1), b)


def obstacle_box_vs_fixed(
    moving_dims, fixed_dims, fixed_center, robot_idx, moving_offset=np.zeros(3)
):
    """Moving box (offset from the tracked grasp point) vs a fixed box.

    Marginalizes the fixed center and embeds the 3D constraint into the 6D slot
    of arm ``robot_idx`` (0 = red [0:3], 1 = blue [3:6])."""
    c6 = c_obstacle_from_box_pair(moving_dims, fixed_dims)
    A, b = c6.A(), c6.b()
    A_move, A_fixed = A[:, :3], A[:, 3:]
    b_new = b - A_move @ np.asarray(moving_offset) - A_fixed @ np.asarray(fixed_center)
    z = np.zeros((A_move.shape[0], 3))
    A6 = np.hstack([A_move, z]) if robot_idx == 0 else np.hstack([z, A_move])
    return pd.HPolyhedron(A6, b_new)


def obstacle_box_vs_box(dims0, dims1, offset0=np.zeros(3), offset1=np.zeros(3)):
    """Two moving boxes (each offset from its arm's grasp point)."""
    c6 = c_obstacle_from_box_pair(dims0, dims1)
    A, b = c6.A(), c6.b()
    b_new = b - A @ np.concatenate([np.asarray(offset0), np.asarray(offset1)])
    return pd.HPolyhedron(A, b_new)


# ===========================================================================
# Per-phase obstacle assembly
# ===========================================================================
def phase_obstacles(is_held, stationary_bricks):
    """Build the obstacle list for one phase.

    ``stationary_bricks`` is a list of ``(name, dims, centroid)`` for bricks
    resting on the pallet or in a bin during this phase. ``is_held`` adds the
    carried-brick pairs (one brick rigidly held by each arm)."""
    obstacles, descs = [], []
    arms = [ARM_RED_PARTS, ARM_BLUE_PARTS]  # arm_idx 0 = red, 1 = blue
    brick_offset = np.array([0.0, 0.0, HELD_BRICK_TCP_OFFSET_Z])

    # gripper part vs robot bases
    for arm_idx, parts in enumerate(arms):
        for name, dims, off in parts:
            for b_name, b_dims, b_center in BASES:
                obstacles.append(obstacle_box_vs_fixed(dims, b_dims, b_center, arm_idx, off))
                descs.append(f"arm{arm_idx}_{name}_vs_{b_name}")

    # gripper fingers vs bin walls (the hand body may briefly clip a wall top)
    for arm_idx, parts in enumerate(arms):
        for name, dims, off in parts:
            if "finger" not in name:
                continue
            for w_name, w_dims, w_center in WALLS:
                obstacles.append(obstacle_box_vs_fixed(dims, w_dims, w_center, arm_idx, off))
                descs.append(f"arm{arm_idx}_{name}_vs_{w_name}")

    # gripper part vs stationary bricks
    for arm_idx, parts in enumerate(arms):
        for name, dims, off in parts:
            for s_name, s_dims, s_center in stationary_bricks:
                obstacles.append(obstacle_box_vs_fixed(dims, s_dims, s_center, arm_idx, off))
                descs.append(f"arm{arm_idx}_{name}_vs_{s_name}")

    # cross-arm gripper vs gripper
    for n_red, d_red, o_red in ARM_RED_PARTS:
        for n_blue, d_blue, o_blue in ARM_BLUE_PARTS:
            obstacles.append(obstacle_box_vs_box(d_red, d_blue, o_red, o_blue))
            descs.append(f"red_{n_red}_vs_blue_{n_blue}")

    if is_held:
        # held brick vs bases
        for arm_idx in (0, 1):
            for b_name, b_dims, b_center in BASES:
                obstacles.append(
                    obstacle_box_vs_fixed(BRICK_DIMS, b_dims, b_center, arm_idx, brick_offset)
                )
                descs.append(f"arm{arm_idx}_held_brick_vs_{b_name}")

        # held brick vs walls
        for arm_idx in (0, 1):
            for w_name, w_dims, w_center in WALLS:
                obstacles.append(
                    obstacle_box_vs_fixed(BRICK_DIMS, w_dims, w_center, arm_idx, brick_offset)
                )
                descs.append(f"arm{arm_idx}_held_brick_vs_{w_name}")

        # held brick vs stationary bricks
        for arm_idx in (0, 1):
            for s_name, s_dims, s_center in stationary_bricks:
                obstacles.append(
                    obstacle_box_vs_fixed(BRICK_DIMS, s_dims, s_center, arm_idx, brick_offset)
                )
                descs.append(f"arm{arm_idx}_held_brick_vs_{s_name}")

        # cross-arm: each arm's gripper part vs the other arm's held brick
        for name, dims, off in ARM_BLUE_PARTS:
            obstacles.append(obstacle_box_vs_box(BRICK_DIMS, dims, brick_offset, off))
            descs.append(f"blue_{name}_vs_red_held_brick")
        for name, dims, off in ARM_RED_PARTS:
            obstacles.append(obstacle_box_vs_box(dims, BRICK_DIMS, off, brick_offset))
            descs.append(f"red_{name}_vs_blue_held_brick")

        # held brick vs held brick
        obstacles.append(obstacle_box_vs_box(BRICK_DIMS, BRICK_DIMS, brick_offset, brick_offset))
        descs.append("held_brick_red_vs_held_brick_blue")

    return obstacles, descs


# ===========================================================================
# Waypoint generators (6D state convention: [red_xyz, blue_xyz])
# ===========================================================================
def _add_to_last(waypoints, delta_red, delta_blue):
    wp = waypoints[-1].copy()
    wp[:3] += delta_red
    wp[3:] += delta_blue
    return wp


def generate_approach_motion(
    start_blue,
    start_red,
    goal_blue,
    goal_red,
    stationary_packages,
    gripper_dims,
    pallet_center,
    pallet_dims,
    finger_below_tcp=FINGER_BELOW_TCP,
    clearance=0.05,
    brick_top_z_fallback=0.21,
):
    """Empty-gripper motion: lift, side-step to ferry lanes, ferry Y, lower.

    ``stationary_packages`` is a list of ``(center, dims)`` used only to size the
    safe lift height above the tallest brick top."""
    waypoints = [np.concatenate([start_red, start_blue])]

    if stationary_packages:
        stationary_top_z = max(c[2] + d[2] / 2 for c, d in stationary_packages)
    else:
        stationary_top_z = brick_top_z_fallback
    safe_z = stationary_top_z + finger_below_tcp + clearance
    waypoints.append(
        _add_to_last(
            waypoints,
            np.array([0, 0, safe_z - waypoints[-1][2]]),
            np.array([0, 0, safe_z - waypoints[-1][5]]),
        )
    )

    x_clearance = 0.025
    pallet_width = pallet_dims[0]
    blue_lane_x = pallet_center[0] - pallet_width / 2 + gripper_dims[0] / 2 + x_clearance
    red_lane_x = pallet_center[0] + pallet_width / 2 - gripper_dims[0] / 2 - x_clearance
    waypoints.append(
        _add_to_last(
            waypoints,
            np.array([red_lane_x - waypoints[-1][0], 0, 0]),
            np.array([blue_lane_x - waypoints[-1][3], 0, 0]),
        )
    )

    waypoints.append(
        _add_to_last(
            waypoints,
            np.array([0, goal_red[1] - waypoints[-1][1], 0]),
            np.array([0, goal_blue[1] - waypoints[-1][4], 0]),
        )
    )

    delta_red_x = goal_red[0] - waypoints[-1][0]
    delta_blue_x = goal_blue[0] - waypoints[-1][3]
    if abs(delta_red_x) > 1e-3 or abs(delta_blue_x) > 1e-3:
        waypoints.append(
            _add_to_last(waypoints, np.array([delta_red_x, 0, 0]), np.array([delta_blue_x, 0, 0]))
        )

    waypoints.append(
        _add_to_last(waypoints, goal_red - waypoints[-1][:3], goal_blue - waypoints[-1][3:])
    )
    return np.array(waypoints)


def generate_forward_motion(
    start_blue,
    start_red,
    goal_blue,
    goal_red,
    blue_brick_dims,
    red_brick_dims,
    stationary_brick_dims,
    gripper_dims,
    pallet_center,
    pallet_dims,
    offload_blue_center,
    offload_red_center,
):
    """Held-gripper motion (pallet -> offload): lift to clear the bin walls,
    side-step to ferry lanes, ferry Y into the bin, diagonal to the drop XY,
    then lower. ``stationary_brick_dims`` sizes the lift height only."""
    waypoints = [np.concatenate([start_red, start_blue])]
    clearance = 0.01

    max_stat_h = np.max([d[2] for d in stationary_brick_dims]) if stationary_brick_dims else 0.0
    required_lift_height = np.max([max_stat_h, OFFLOAD_WALL_HEIGHT]) + clearance
    waypoints.append(
        _add_to_last(
            waypoints,
            np.array([0, 0, required_lift_height]),
            np.array([0, 0, required_lift_height]),
        )
    )

    x_clearance = 0.025
    pallet_width = pallet_dims[0]
    blue_lane_x = (
        pallet_center[0]
        - pallet_width / 2
        + np.max([gripper_dims[0], blue_brick_dims[0]]) / 2
        + x_clearance
    )
    red_lane_x = (
        pallet_center[0]
        + pallet_width / 2
        - np.max([gripper_dims[0], red_brick_dims[0]]) / 2
        - x_clearance
    )
    delta_x_red_ferry = red_lane_x - waypoints[-1][0]
    delta_x_blue_ferry = blue_lane_x - waypoints[-1][3]

    # avoid grippers intersecting when sliding to the ferry lanes (crossed arms)
    y_separation = np.abs(waypoints[-1][1] - waypoints[-1][4])
    arms_crossed = waypoints[-1][3] > waypoints[-1][0]
    y_thresh = np.max([gripper_dims[1], blue_brick_dims[1] / 2 + red_brick_dims[1] / 2])
    if y_separation < y_thresh + 0.02 and arms_crossed:
        delta_to_clear = y_thresh + 0.02 - y_separation
        if waypoints[-1][1] < waypoints[-1][4]:
            clearing_delta = np.array([0, -delta_to_clear, 0, 0, 0, 0])
        else:
            clearing_delta = np.array([0, 0, 0, 0, -delta_to_clear, 0])
        waypoints.append(_add_to_last(waypoints, clearing_delta[:3], clearing_delta[3:]))

    waypoints.append(
        _add_to_last(
            waypoints, np.array([delta_x_red_ferry, 0, 0]), np.array([delta_x_blue_ferry, 0, 0])
        )
    )

    target_y_blue = (
        offload_blue_center[1]
        + OFFLOAD_DIMS[1] / 2
        - np.max([gripper_dims[1], blue_brick_dims[1]]) / 2
        - 0.02
    )
    target_y_red = (
        offload_red_center[1]
        + OFFLOAD_DIMS[1] / 2
        - np.max([gripper_dims[1], red_brick_dims[1]]) / 2
        - 0.02
    )
    waypoints.append(
        _add_to_last(
            waypoints,
            np.array([0, target_y_red - waypoints[-1][1], 0]),
            np.array([0, target_y_blue - waypoints[-1][4], 0]),
        )
    )

    delta_xy_red = goal_red[:2] - waypoints[-1][:2]
    delta_xy_blue = goal_blue[:2] - waypoints[-1][3:5]
    waypoints.append(
        _add_to_last(
            waypoints,
            np.array([delta_xy_red[0], delta_xy_red[1], 0]),
            np.array([delta_xy_blue[0], delta_xy_blue[1], 0]),
        )
    )

    waypoints.append(
        _add_to_last(waypoints, goal_red - waypoints[-1][:3], goal_blue - waypoints[-1][3:])
    )
    return np.array(waypoints)


# ===========================================================================
# Bin walls (pallet + grey offload + red offload, 4 each = 12)
# ===========================================================================
WALL_HEIGHT = 0.85 * BRICK_DIMS[2]  # 0.1785
WALL_THICK = 0.02
WALL_OFFSET_X = 0.03 + 0.015 + WALL_THICK / 2 + 0.02  # 0.075
WALL_OFFSET_Y = 0.080 + WALL_THICK / 2 + 0.02  # 0.110


def _brick_aabb(centers_xy):
    bw, bd = BRICK_DIMS[0] / 2.0, BRICK_DIMS[1] / 2.0
    xs = [c[0] for c in centers_xy]
    ys = [c[1] for c in centers_xy]
    return min(xs) - bw, max(xs) + bw, min(ys) - bd, max(ys) + bd


def _walls_for_zone(zone_name, brick_aabb):
    xmin, xmax, ymin, ymax = brick_aabb
    wx_lo, wx_hi = xmin - WALL_OFFSET_X, xmax + WALL_OFFSET_X
    wy_lo, wy_hi = ymin - WALL_OFFSET_Y, ymax + WALL_OFFSET_Y
    wz = WALL_HEIGHT / 2.0
    len_ns_x = (wx_hi - wx_lo) + WALL_THICK
    len_ew_y = (wy_hi - wy_lo) + WALL_THICK
    return [
        (
            f"wall_{zone_name}_N",
            np.array([len_ns_x, WALL_THICK, WALL_HEIGHT]),
            np.array([(wx_lo + wx_hi) / 2, wy_hi, wz]),
        ),
        (
            f"wall_{zone_name}_S",
            np.array([len_ns_x, WALL_THICK, WALL_HEIGHT]),
            np.array([(wx_lo + wx_hi) / 2, wy_lo, wz]),
        ),
        (
            f"wall_{zone_name}_E",
            np.array([WALL_THICK, len_ew_y, WALL_HEIGHT]),
            np.array([wx_hi, (wy_lo + wy_hi) / 2, wz]),
        ),
        (
            f"wall_{zone_name}_W",
            np.array([WALL_THICK, len_ew_y, WALL_HEIGHT]),
            np.array([wx_lo, (wy_lo + wy_hi) / 2, wz]),
        ),
    ]


def build_all_walls():
    pallet_aabb = _brick_aabb([(c[0], c[1]) for c in BRICK_CENTROIDS])
    grey_aabb = _brick_aabb(
        [
            (OFFLOAD_GREY[0] - SLOT_OFFSET_X, OFFLOAD_GREY[1]),
            (OFFLOAD_GREY[0] + SLOT_OFFSET_X, OFFLOAD_GREY[1]),
        ]
    )
    red_aabb = _brick_aabb(
        [
            (OFFLOAD_RED[0] - SLOT_OFFSET_X, OFFLOAD_RED[1]),
            (OFFLOAD_RED[0] + SLOT_OFFSET_X, OFFLOAD_RED[1]),
        ]
    )
    return (
        _walls_for_zone("pallet", pallet_aabb)
        + _walls_for_zone("grey", grey_aabb)
        + _walls_for_zone("red", red_aabb)
    )


WALLS = build_all_walls()


# ===========================================================================
# Drop positions + per-phase brick state
# ===========================================================================
def get_drop_pos(brick_id):
    """Drop xyz for a brick: color picks the pad, move picks the slot (outer for
    move-1, inner for move-2), ±SLOT_OFFSET_X in X around the pad center."""
    color = BRICK_COLOR[brick_id]
    is_move_1 = brick_id in (M1_BLUE, M1_RED)
    base = OFFLOAD_GREY if color == "grey" else OFFLOAD_RED
    outer_sign = -1.0 if color == "grey" else +1.0
    sign = outer_sign if is_move_1 else -outer_sign
    return base + np.array([sign * SLOT_OFFSET_X, 0.0, 0.0])


def drop_centroid(brick_id):
    p = get_drop_pos(brick_id)
    return np.array([p[0], p[1], BRICK_CENTROID_Z])


def offload_center(brick_id):
    return OFFLOAD_GREY if BRICK_COLOR[brick_id] == "grey" else OFFLOAD_RED


def named_bricks(specs):
    """``specs`` is a list of (brick_id, centroid) -> (name, dims, centroid)."""
    return [(f"brick_{bid}", BRICK_DIMS, c) for bid, c in specs]


# ===========================================================================
# Per-phase planning
# ===========================================================================
def make_domain(z_floor):
    lb = np.array([-0.6, -0.6, z_floor, -0.6, -0.6, z_floor])
    ub = np.array([1.7, 0.8, 0.9, 1.7, 0.8, 0.9])
    return pd.HPolyhedron.MakeBox(lb, ub)


def project_into_domain(waypoints, domain):
    """Clip each waypoint into the box domain (the box projection)."""
    A, b = domain.A(), domain.b()
    # MakeBox rows are ±I, so recover lb/ub directly.
    lb = np.full(waypoints.shape[1], -np.inf)
    ub = np.full(waypoints.shape[1], np.inf)
    for row, rhs in zip(A, b):
        idx = int(np.argmax(np.abs(row)))
        if row[idx] > 0:
            ub[idx] = min(ub[idx], rhs / row[idx])
        else:
            lb[idx] = max(lb[idx], rhs / row[idx])
    return np.clip(waypoints, lb, ub)


def plan_phase(name, waypoints, is_held, stationary_bricks, planner, vel_set, acc_set):
    domain = make_domain(TRAJ_GRASP_Z if is_held else FINGER_BELOW_TCP)
    waypoints = project_into_domain(np.asarray(waypoints, dtype=float), domain)
    obstacles, descs = phase_obstacles(is_held, stationary_bricks)

    print(
        f"\n=== Phase {name}: {len(waypoints)} waypoints, {len(obstacles)} "
        f"obstacles, {'HELD' if is_held else 'EMPTY'} ==="
    )

    t0 = time.perf_counter()
    if planner == "bmtp":
        res = MinimumTimePlanner(
            rel_term=0.01,
            max_iter=120,
            collision_check_tol=1e-3,
            trajectory_to_plane_tol=1e-3,
        ).solve(waypoints, obstacles, domain, Limits(vel_set, acc_set), verbose=False)
        extra = f"{res.num_iterations} iters"
    else:
        res = EISCSPlanner(rel_term=0.01).solve(
            waypoints, obstacles, domain, Limits(vel_set, acc_set), verbose=False
        )
        extra = f"{res.num_regions} regions"
    solve_s = time.perf_counter() - t0

    hits = intersect_with_hpolyhedra(res.trajectory, obstacles, tol=1e-3)
    free = hits == {}
    print(f"  T={res.total_time:.2f}s ({extra}), solve {solve_s:.2f}s, " f"collision-free={free}")
    if not free:
        for seg_id, obs_ids in hits.items():
            for oid in obs_ids:
                print(
                    f"    seg {seg_id} hits obs {oid}: "
                    f"{descs[oid] if oid < len(descs) else '?'}"
                )
    return res.trajectory, res.total_time, solve_s, waypoints


# ===========================================================================
# Rendering
# ===========================================================================
def render(path_out, phases, planner, total_T):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, (ax_top, ax_z) = plt.subplots(1, 2, figsize=(15, 7))

    # --- top view (x-y) ----------------------------------------------------
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("x [m]")
    ax_top.set_ylabel("y [m]")
    ax_top.set_title("Top view (TCP paths)")

    for _, _, w_dims, w_center in [("", *w) for w in WALLS]:
        lb = w_center[:2] - w_dims[:2] / 2
        ax_top.add_patch(
            mpatches.Rectangle(
                lb, w_dims[0], w_dims[1], facecolor="0.4", edgecolor="none", alpha=0.6, zorder=2
            )
        )
    for c in BRICK_CENTROIDS:
        ax_top.add_patch(
            mpatches.Rectangle(
                c[:2] - BRICK_DIMS[:2] / 2,
                BRICK_DIMS[0],
                BRICK_DIMS[1],
                facecolor="tan",
                edgecolor="k",
                alpha=0.7,
                zorder=3,
            )
        )
    for bid in range(4):
        p = get_drop_pos(bid)
        col = "0.6" if BRICK_COLOR[bid] == "grey" else "lightcoral"
        ax_top.add_patch(
            mpatches.Rectangle(
                p[:2] - BRICK_DIMS[:2] / 2,
                BRICK_DIMS[0],
                BRICK_DIMS[1],
                facecolor=col,
                edgecolor="k",
                alpha=0.7,
                zorder=3,
            )
        )

    t_cum = 0.0
    z_segments = []
    for name, traj, T, _ in phases:
        ts = np.linspace(traj.initial_time, traj.final_time, 200)
        pts = np.array([np.asarray(traj(t)).reshape(-1) for t in ts])
        ax_top.plot(pts[:, 0], pts[:, 1], color="tab:red", lw=2, zorder=5)
        ax_top.plot(pts[:, 3], pts[:, 4], color="tab:blue", lw=2, zorder=5)
        z_segments.append((t_cum, np.linspace(0, T, len(ts)), pts))
        t_cum += T

    ax_top.plot([], [], color="tab:red", lw=2, label="red TCP")
    ax_top.plot([], [], color="tab:blue", lw=2, label="blue TCP")
    ax_top.legend(loc="upper right")

    # --- z profile vs time -------------------------------------------------
    ax_z.set_xlabel("time [s]")
    ax_z.set_ylabel("z [m]")
    ax_z.set_title("Height profile")
    for t0, ts_local, pts in z_segments:
        ax_z.plot(t0 + ts_local, pts[:, 2], color="tab:red", lw=2)
        ax_z.plot(t0 + ts_local, pts[:, 5], color="tab:blue", lw=2)
    for t0, _, _ in z_segments[1:]:
        ax_z.axvline(t0, color="0.7", ls="--", lw=1)
    ax_z.axhline(TRAJ_GRASP_Z, color="0.5", ls=":", lw=1)

    fig.suptitle(
        f"Dual-arm full unload ({planner.upper()}): "
        f"total T = {total_T:.2f}s over {len(phases)} phases"
    )
    fig.tight_layout()
    fig.savefig(path_out, dpi=130)
    plt.close(fig)


# ===========================================================================
# Meshcat animation of the collision geometry
# ===========================================================================
def animate(meshcat, phases, held_assignment):
    """``held_assignment[i]`` maps phase index -> {arm: brick_id} for carried
    bricks; bricks not currently held are drawn at their static centroid."""

    def box(name, dims, color):
        meshcat.SetObject(name, pd.Box(*dims), rgba=color)

    for name, dims, center in WALLS:
        box(f"scene/{name}", dims, pd.Rgba(0.55, 0.40, 0.20, 0.85))
        meshcat.SetTransform(f"scene/{name}", pd.RigidTransform(center))
    for name, dims, center in BASES:
        box(f"scene/{name}", dims, pd.Rgba(0.0, 0.8, 0.0, 0.35))
        meshcat.SetTransform(f"scene/{name}", pd.RigidTransform(center))

    brick_rgba = {
        bid: (
            pd.Rgba(0.55, 0.55, 0.55, 1.0)
            if BRICK_COLOR[bid] == "grey"
            else pd.Rgba(0.85, 0.10, 0.10, 1.0)
        )
        for bid in range(4)
    }
    for bid in range(4):
        box(f"scene/brick_{bid}", BRICK_DIMS, brick_rgba[bid])
    for arm in ("red", "blue"):
        for pname, dims, _ in ARM_RED_PARTS:
            box(f"scene/grip_{arm}_{pname}", dims, pd.Rgba(1.0, 0.5, 0.0, 0.35))

    parts = {"red": ARM_RED_PARTS, "blue": ARM_BLUE_PARTS}
    brick_static = {bid: BRICK_CENTROIDS[bid].copy() for bid in range(4)}

    meshcat.StartRecording()
    t_cum = 0.0
    for pi, (name, traj, T, _) in enumerate(phases):
        held = held_assignment[pi]
        ts = np.linspace(traj.initial_time, traj.final_time, max(2, int(T * 30)))
        for t in ts:
            s = np.asarray(traj(t)).reshape(-1)
            tcp = {"red": s[:3], "blue": s[3:]}
            t_global = t_cum + (t - traj.initial_time)
            for arm in ("red", "blue"):
                for pname, dims, off in parts[arm]:
                    meshcat.SetTransform(
                        f"scene/grip_{arm}_{pname}",
                        pd.RigidTransform(tcp[arm] + off),
                        time_in_recording=t_global,
                    )
            for bid in range(4):
                pos = brick_static[bid]
                for arm, hid in held.items():
                    if hid == bid:
                        pos = tcp[arm] + np.array([0, 0, HELD_BRICK_TCP_OFFSET_Z])
                meshcat.SetTransform(
                    f"scene/brick_{bid}", pd.RigidTransform(pos), time_in_recording=t_global
                )
        for arm, hid in held.items():
            brick_static[hid] = drop_centroid(hid)
        t_cum += T
    meshcat.StopRecording()
    meshcat.PublishRecording()


# ===========================================================================
# main
# ===========================================================================
def main(planner="bmtp", show_meshcat=True):
    sphere_v = pd.Hyperellipsoid.MakeHypersphere(V_MAX, np.zeros(3))
    sphere_a = pd.Hyperellipsoid.MakeHypersphere(A_MAX, np.zeros(3))
    vel_set = pd.CartesianProduct(sphere_v, sphere_v)
    acc_set = pd.CartesianProduct(sphere_a, sphere_a)

    stat_AB = named_bricks([(M2_BLUE, BRICK_CENTROIDS[M2_BLUE]), (M2_RED, BRICK_CENTROIDS[M2_RED])])
    stat_C = named_bricks(
        [
            (M2_BLUE, BRICK_CENTROIDS[M2_BLUE]),
            (M2_RED, BRICK_CENTROIDS[M2_RED]),
            (M1_BLUE, drop_centroid(M1_BLUE)),
            (M1_RED, drop_centroid(M1_RED)),
        ]
    )
    stat_D = named_bricks([(M1_BLUE, drop_centroid(M1_BLUE)), (M1_RED, drop_centroid(M1_RED))])
    stat_E = named_bricks([(bid, drop_centroid(bid)) for bid in range(4)])

    # Phase A: home -> grasp move-1 bricks (empty)
    wp_A = generate_approach_motion(
        start_blue=P_BLUE_QREF,
        start_red=P_RED_QREF,
        goal_blue=BRICK_GRASP_POINTS[M1_BLUE],
        goal_red=BRICK_GRASP_POINTS[M1_RED],
        stationary_packages=[
            (BRICK_CENTROIDS[M2_BLUE], BRICK_DIMS),
            (BRICK_CENTROIDS[M2_RED], BRICK_DIMS),
        ],
        gripper_dims=GRIPPER_DIMS_PLANNER,
        pallet_center=PALLET_CENTER,
        pallet_dims=PALLET_DIMS,
    )

    # Phase B: grasp move-1 -> offload (held)
    wp_B = generate_forward_motion(
        start_blue=BRICK_GRASP_POINTS[M1_BLUE],
        start_red=BRICK_GRASP_POINTS[M1_RED],
        goal_blue=get_drop_pos(M1_BLUE),
        goal_red=get_drop_pos(M1_RED),
        blue_brick_dims=BRICK_DIMS,
        red_brick_dims=BRICK_DIMS,
        stationary_brick_dims=[BRICK_DIMS, BRICK_DIMS],
        gripper_dims=GRIPPER_DIMS_PLANNER,
        pallet_center=PALLET_CENTER,
        pallet_dims=PALLET_DIMS,
        offload_blue_center=offload_center(M1_BLUE),
        offload_red_center=offload_center(M1_RED),
    )

    # Phase C: offload -> grasp move-2 bricks (empty)
    wp_C = generate_approach_motion(
        start_blue=get_drop_pos(M1_BLUE),
        start_red=get_drop_pos(M1_RED),
        goal_blue=BRICK_GRASP_POINTS[M2_BLUE],
        goal_red=BRICK_GRASP_POINTS[M2_RED],
        stationary_packages=[
            (BRICK_CENTROIDS[M2_BLUE], BRICK_DIMS),
            (BRICK_CENTROIDS[M2_RED], BRICK_DIMS),
            (drop_centroid(M1_BLUE), BRICK_DIMS),
            (drop_centroid(M1_RED), BRICK_DIMS),
        ],
        gripper_dims=GRIPPER_DIMS_PLANNER,
        pallet_center=PALLET_CENTER,
        pallet_dims=PALLET_DIMS,
    )

    # Phase D: grasp move-2 -> offload (held)
    wp_D = generate_forward_motion(
        start_blue=BRICK_GRASP_POINTS[M2_BLUE],
        start_red=BRICK_GRASP_POINTS[M2_RED],
        goal_blue=get_drop_pos(M2_BLUE),
        goal_red=get_drop_pos(M2_RED),
        blue_brick_dims=BRICK_DIMS,
        red_brick_dims=BRICK_DIMS,
        stationary_brick_dims=[BRICK_DIMS, BRICK_DIMS],
        gripper_dims=GRIPPER_DIMS_PLANNER,
        pallet_center=PALLET_CENTER,
        pallet_dims=PALLET_DIMS,
        offload_blue_center=offload_center(M2_BLUE),
        offload_red_center=offload_center(M2_RED),
    )

    # Phase E: offload -> home (empty)
    wp_E = generate_approach_motion(
        start_blue=get_drop_pos(M2_BLUE),
        start_red=get_drop_pos(M2_RED),
        goal_blue=P_BLUE_QREF,
        goal_red=P_RED_QREF,
        stationary_packages=[(drop_centroid(bid), BRICK_DIMS) for bid in range(4)],
        gripper_dims=GRIPPER_DIMS_PLANNER,
        pallet_center=PALLET_CENTER,
        pallet_dims=PALLET_DIMS,
    )

    specs = [
        ("A (approach M1)", wp_A, False, stat_AB),
        ("B (forward M1)", wp_B, True, stat_AB),
        ("C (approach M2)", wp_C, False, stat_C),
        ("D (forward M2)", wp_D, True, stat_D),
        ("E (return home)", wp_E, False, stat_E),
    ]
    held_assignment = [
        {},
        {"blue": M1_BLUE, "red": M1_RED},
        {},
        {"blue": M2_BLUE, "red": M2_RED},
        {},
    ]

    phases = []
    total_T = 0.0
    total_solve = 0.0
    for name, wp, is_held, stationary in specs:
        traj, T, solve_s, wp_proj = plan_phase(
            name, wp, is_held, stationary, planner, vel_set, acc_set
        )
        phases.append((name, traj, T, wp_proj))
        total_T += T
        total_solve += solve_s

    print(
        f"\n=== Total: T = {total_T:.2f}s, solve = {total_solve:.2f}s " f"({planner.upper()}) ==="
    )

    out_dir = pathlib.Path(__file__).resolve().parent / "_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"dual_arm_unload_{planner}.png"
    render(out, phases, planner, total_T)
    print(f"wrote {out}")

    if show_meshcat:
        meshcat = pd.StartMeshcat()
        animate(meshcat, phases, held_assignment)
        print(f"meshcat: {meshcat.web_url()}  (Ctrl-C to exit)")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner", choices=["bmtp", "scs"], default="bmtp", help="planner backend (default: bmtp)"
    )
    parser.add_argument(
        "--no-meshcat", action="store_true", help="skip the meshcat animation (run headless)"
    )
    args = parser.parse_args()
    main(planner=args.planner, show_meshcat=not args.no_meshcat)
