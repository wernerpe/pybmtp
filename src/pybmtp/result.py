"""Typed result objects returned by the planners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pybezier as pb


@dataclass
class SolveResult:
    """Outcome of a minimum-time solve.

    Attributes
    ----------
    trajectory:
        The final time-scaled trajectory (physical seconds).
    segment_time:
        Per-segment duration of the final trajectory.
    total_time:
        Total trajectory duration (``segment_time * num_segments``).
    num_iterations:
        Number of biconvex outer iterations performed.
    converged:
        True if the loop terminated on the relative-improvement criterion
        rather than hitting ``max_iter``.
    planes:
        Final separating planes, ``{(segment_id, obstacle_id): (a_curve, b_curve)}``.
    feasible_trajectories / feasible_segment_times / feasible_wall_times:
        The collision-free iterates found along the way (normalized-time curves),
        their per-segment times, and the wall-clock time (s) at which each was
        found. Useful for plotting convergence.
    timing:
        Wall-clock breakdown of the solve (keys: ``total``, ``trajectory_update``,
        ``trajectory_solver``, ``collision_check``, ``plane_update``).
    """

    trajectory: pb.CompositeBezierCurve
    segment_time: float
    total_time: float
    num_iterations: int
    converged: bool
    planes: Dict[Tuple[int, int], Tuple[pb.BezierCurve, pb.BezierCurve]] = field(
        default_factory=dict
    )
    feasible_trajectories: List[pb.CompositeBezierCurve] = field(default_factory=list)
    feasible_segment_times: List[float] = field(default_factory=list)
    feasible_wall_times: List[float] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=dict)
