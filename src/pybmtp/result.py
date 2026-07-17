"""Typed result objects returned by the planners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pybezier as pb
import pydrake.all as pd


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


@dataclass
class SCSResult:
    """Outcome of an :class:`~pybmtp.scs.EISCSPlanner` solve.

    Attributes
    ----------
    trajectory:
        The final time-scaled trajectory (physical seconds).
    total_time:
        Total trajectory duration.
    regions:
        The collision-free convex regions the trajectory was planned through
        (after near-zero-length pruning and Assumption-1 splitting).
    num_regions:
        Number of regions actually solved through (post-pruning).
    solver_type:
        Which scsplanning backend produced the trajectory (``"biconvex"``,
        ``"polygonal"``, or ``"nonconvex"``).
    timing:
        Wall-clock breakdown (keys: ``total``, ``region_generation``,
        ``trajectory_optimization``).
    """

    trajectory: pb.CompositeBezierCurve
    total_time: float
    regions: List[pd.HPolyhedron] = field(default_factory=list)
    num_regions: int = 0
    solver_type: str = "biconvex"
    timing: Dict[str, float] = field(default_factory=dict)
