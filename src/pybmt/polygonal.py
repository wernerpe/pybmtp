"""Polygonal warm-start initializer.

Given a sequence of waypoints, this connects consecutive pairs with
minimum-time Bezier segments that respect velocity and acceleration limits,
producing a constraint-feasible polygonal trajectory. It is the warm start that
seeds the biconvex planners (:class:`~pybmt.MinimumTimePlanner`,
:class:`~pybmt.EISCSPlanner`).

Each segment is a straight line in space, so the per-segment problem collapses
to a 1D minimum-time program along the segment direction: project the velocity
and acceleration sets onto the direction to get scalar limits, then solve a
small rotated-cone program for the segment duration and 1D control points.
"""

from __future__ import annotations

import numpy as np
import pybezier as pb
import pydrake.all as pd

from .limits import Limits

__all__ = [
    "PolygonalInitializer",
    "solve_segment_time_optimal",
    "compute_set_limit",
]


class PolygonalInitializer:
    """Connects waypoints with constraint-feasible minimum-time Bezier segments.

    The straight-line warm start uses the velocity and acceleration limits only
    (each segment collapses to a 1D minimum-time program along its direction);
    any jerk / snap limits carried by ``limits`` are enforced later by the
    biconvex trajectory update, not here.

    Parameters
    ----------
    limits:
        :class:`~pybmt.limits.Limits`; its velocity and acceleration sets bound
        the warm-start segments.
    trajectory_degree:
        Bezier degree of each segment. Must be >= 5 to zero the boundary
        acceleration (needed for C2-continuous downstream planning).
    force_same_segment_time:
        If True, every segment is given the same duration equal to the slowest
        segment's optimal time. This yields a uniform time grid, which some
        downstream solvers prefer as a warm start.
    zero_boundary_acceleration:
        If True (and degree >= 5), constrain start/end acceleration of every
        segment to zero.
    """

    def __init__(
        self,
        limits: Limits,
        trajectory_degree: int = 5,
        force_same_segment_time: bool = False,
        zero_boundary_acceleration: bool = True,
    ):
        self.limits = limits
        self.velocity_set = limits.velocity
        self.acceleration_set = limits.acceleration
        self.trajectory_degree = trajectory_degree
        self.force_same_segment_time = force_same_segment_time
        self.zero_boundary_acceleration = zero_boundary_acceleration

    def initialize(self, waypoints: np.ndarray) -> pb.CompositeBezierCurve:
        """Build a time-contiguous composite Bezier curve through ``waypoints``.

        ``waypoints`` is an ``(N, dim)`` array of corner points.
        """
        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2 or len(waypoints) < 2:
            raise ValueError("waypoints must be an (N, dim) array with N >= 2")

        curves = []
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            curve = solve_segment_time_optimal(
                start,
                end,
                self.velocity_set,
                self.acceleration_set,
                self.trajectory_degree,
                zero_boundary_acceleration=self.zero_boundary_acceleration,
            )
            if curve is None:
                raise RuntimeError(f"failed to solve minimum-time segment {start} -> {end}")
            curves.append(curve)

        if self.force_same_segment_time:
            max_duration = max(c.final_time - c.initial_time for c in curves)
            curves = [pb.BezierCurve(c.points, 0.0, max_duration) for c in curves]

        # Stitch segments into a single contiguous timeline.
        shifted = []
        t = 0.0
        for c in curves:
            duration = c.final_time - c.initial_time
            shifted.append(pb.BezierCurve(c.points, t, t + duration))
            t += duration
        return pb.CompositeBezierCurve(shifted)

    __call__ = initialize


def solve_segment_time_optimal(
    start: np.ndarray,
    end: np.ndarray,
    velocity_set: pd.ConvexSet,
    acceleration_set: pd.ConvexSet,
    degree: int,
    zero_boundary_acceleration: bool = True,
) -> pb.BezierCurve | None:
    # TODO(pete): Change this from the SCS Polygonal initializer to the BMTP trajupdate with no planes

    """Minimum-time Bezier segment along the straight line ``start`` -> ``end``.

    Solves the 1D normalized minimum-time program (paper Eq. 17) for the scalar
    progress ``rho(s)`` along the segment, then maps the optimal 1D controls
    back into space. Returns ``None`` if the convex program is infeasible.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    dist = float(np.linalg.norm(direction))

    # Degenerate (duplicate) waypoints: emit a trivial near-zero-duration point.
    if dist < 1e-6:
        pts = np.tile(start.reshape(1, -1), (degree + 1, 1))
        return pb.BezierCurve(pts, 0.0, 1e-6)

    u_dir = direction / dist

    # Project the (multi-dim) velocity/acceleration sets onto the 1D direction.
    v_max = compute_set_limit(u_dir, velocity_set)
    a_max = compute_set_limit(u_dir, acceleration_set)  # forward (accelerate)
    a_min = -compute_set_limit(-u_dir, acceleration_set)  # backward (brake)

    prog = pd.MathematicalProgram()

    # T   : segment duration.
    # S   : inverse duration (1/T), related to T by the rotated-cone T*S >= 1.
    # rho : 1D control points of the normalized progress r(s), s in [0, 1].
    T = prog.NewContinuousVariables(1, "T")[0]
    S = prog.NewContinuousVariables(1, "S")[0]
    rho = prog.NewContinuousVariables(degree + 1, "rho")

    # T * S >= 1 via the rotated Lorentz cone z0*z1 >= z2^2 with z = [T, S, 1].
    prog.AddRotatedLorentzConeConstraint(
        A=np.array([[1, 0], [0, 1], [0, 0]]),
        b=np.array([0, 0, 1]),
        vars=[T, S],
    )
    prog.AddBoundingBoxConstraint(1e-3, np.inf, T)
    prog.AddBoundingBoxConstraint(0, np.inf, S)

    # Endpoints of the normalized path: rho_0 = 0, rho_K = S * dist.
    prog.AddLinearEqualityConstraint(rho[0] == 0)
    prog.AddLinearEqualityConstraint(rho[-1] - S * dist == 0)

    # Monotonic progress (aids numerical conditioning).
    for i in range(degree):
        prog.AddLinearConstraint(rho[i + 1] >= rho[i])

    # Finite-difference derivatives of the Bezier control points.
    rho_dot = degree * (rho[1:] - rho[:-1])
    rho_ddot = (degree - 1) * (rho_dot[1:] - rho_dot[:-1])

    # Velocity: 0 <= rho_dot <= v_max, with zero boundary velocity.
    for k in range(len(rho_dot)):
        prog.AddLinearConstraint(rho_dot[k] <= v_max)
    prog.AddLinearEqualityConstraint(rho_dot[0] == 0)
    prog.AddLinearEqualityConstraint(rho_dot[-1] == 0)

    # Zero boundary acceleration (needed for C2-continuous stitching).
    if zero_boundary_acceleration and degree >= 5:
        prog.AddLinearEqualityConstraint(rho_ddot[0] == 0)
        prog.AddLinearEqualityConstraint(rho_ddot[-1] == 0)

    # Acceleration: a_min * T <= rho_ddot <= a_max * T (paper Eq. 17f).
    for k in range(len(rho_ddot)):
        prog.AddLinearConstraint(rho_ddot[k] <= a_max * T)
        prog.AddLinearConstraint(rho_ddot[k] >= a_min * T)

    prog.AddLinearCost(T)

    result = pd.ClarabelSolver().Solve(prog)
    if not result.is_success():
        return None

    T_opt = result.GetSolution(T)
    rho_opt = result.GetSolution(rho)

    # De-normalize: q_1d = rho / S = rho * T, then lift to space.
    q_1d = rho_opt * T_opt
    ctrl_pts = start + np.outer(q_1d, u_dir)
    return pb.BezierCurve(ctrl_pts, 0.0, T_opt)


def compute_set_limit(direction: np.ndarray, convex_set: pd.ConvexSet) -> float:
    """Largest ``alpha`` such that ``alpha * direction`` lies in ``convex_set``.

    For an HPolyhedron this is the analytic ray-cast
    ``alpha = min_i b[i] / (A[i] . direction)`` over rows with positive
    projection, which avoids the bisection loop. Other convex sets fall back to
    20 bisection iterations (~1e-6 precision).
    """
    if isinstance(convex_set, pd.HPolyhedron):
        A = convex_set.A()
        b = convex_set.b()
        Ad = A @ direction
        mask = Ad > 1e-12
        if not np.any(mask):
            return 1000.0  # unbounded along this direction
        return float(np.min(b[mask] / Ad[mask]))

    low, high = 0.0, 1000.0
    if convex_set.PointInSet(high * direction):
        return high
    for _ in range(20):
        mid = 0.5 * (low + high)
        if convex_set.PointInSet(mid * direction):
            low = mid
        else:
            high = mid
    return low
