"""The SCS minimum-time planner and its one-shot wrapper.

This is the baseline used in the SCS+EI paper: rather than the biconvex
trajectory/plane alternation of :class:`~pybmtp.MinimumTimePlanner`, it builds a
fixed sequence of overlapping collision-free convex regions (one per path
segment, via exact convex separation in :mod:`eiscs.regions`) and then solves
for the fastest Bezier trajectory through that corridor with the ``scsplanning``
library.

Two preprocessing steps make the corridor safe for ``scsplanning``:

  1. **Near-zero-length pruning.** ``scsplanning``'s crossing-time bisection
     fails on segments of (near) zero length, so regions whose shortest-path
     edge collapses are dropped.
  2. **Assumption-1 splitting** (:func:`eiscs.regions.add_segment_splitting_hyperplanes`):
     non-adjacent regions are cut apart so ``region_i`` and ``region_{i+2}`` no
     longer intersect, as ``scsplanning`` requires.

The exact obstacle-separation region construction means this planner depends
only on ``drake`` + ``pybezier`` + ``scsplanning`` — no GPU / EI-ZO / csdecomp.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import pybezier as pb
import pydrake.all as pd

from pybmtp import Limits

from .regions import add_segment_splitting_hyperplanes, construct_regions_from_obstacles
from .result import SCSResult


def _as_float64_curve(traj: pb.CompositeBezierCurve) -> pb.CompositeBezierCurve:
    """Re-cast a trajectory's control points to ``float64``.

    ``scsplanning`` assembles each segment from Drake-evaluated values, leaving
    the control points as a NumPy ``object`` array. Evaluating such a curve
    returns ``object`` arrays that break downstream NumPy ufuncs (``isfinite``,
    ``allclose``, ...). Rebuilding each segment with explicit ``float64`` points
    fixes this while keeping the public type a genuine ``CompositeBezierCurve``.
    """
    return pb.CompositeBezierCurve(
        [
            pb.BezierCurve(np.asarray(c.points, dtype=float), c.initial_time, c.final_time)
            for c in traj.curves
        ]
    )


def _biconvex_with_pruning(
    q_init: np.ndarray,
    q_term: np.ndarray,
    regions: List[pd.HPolyhedron],
    velocity_set: pd.ConvexSet,
    acceleration_set: pd.ConvexSet,
    trajectory_degree: int,
    path: np.ndarray,
    rel_term: float,
    split_margin: float,
    min_edge_length: float,
    verbose: bool,
) -> Tuple[pb.CompositeBezierCurve, List[pd.HPolyhedron]]:
    """Prune near-zero-length regions, add splitting planes, then run biconvex."""
    from scsplanning import biconvex
    from scsplanning.polygonal import get_points

    # get_points solves the SOCP for the shortest polyline through the regions;
    # consecutive points that nearly coincide would break the crossing-time
    # bisection, so drop those regions.
    points = get_points(q_init, q_term, regions)
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    surviving = [(i, r) for i, (r, d) in enumerate(zip(regions, dists)) if d > min_edge_length]
    original_indices = [i for i, _ in surviving]
    pruned = [r for _, r in surviving]

    if verbose and len(pruned) != len(regions):
        print(
            f"[eiscs] pruned {len(regions) - len(pruned)} near-zero "
            f"region(s); {len(pruned)} remaining"
        )

    if len(pruned) >= 3:
        pruned = add_segment_splitting_hyperplanes(
            path, pruned, original_indices, split_margin=split_margin
        )

    trajectory = biconvex(
        q_init, q_term, pruned, velocity_set, acceleration_set, trajectory_degree, tol=rel_term
    )
    return trajectory, pruned


class EISCSPlanner:
    """Minimum-time Bezier planner via edge-inflation regions + ``scsplanning``.

    Configuration is set once at construction; :meth:`solve` runs region
    construction and the trajectory optimization for a specific instance.

    Parameters
    ----------
    trajectory_degree:
        Bezier degree of each trajectory segment.
    solver_type:
        ``scsplanning`` backend: ``"biconvex"`` (default; the SCS+EI baseline),
        ``"polygonal"`` (shortest-path warm start), or ``"nonconvex"`` (single
        nonconvex solve).
    rel_term:
        Relative termination tolerance for the biconvex solve.
    margin:
        Safety margin pushed toward the obstacle when separating each segment.
    split_margin:
        Half-width of the strip removed from both regions at each Assumption-1
        cut. Must be smaller than half the shortest middle-segment length.
    min_edge_length:
        Shortest-path edge length below which a region is pruned.
    """

    def __init__(
        self,
        trajectory_degree: int = 6,
        solver_type: str = "biconvex",
        rel_term: float = 0.01,
        margin: float = 1e-3,
        split_margin: float = 1e-4,
        min_edge_length: float = 1e-4,
    ):
        if solver_type not in ("biconvex", "polygonal", "nonconvex"):
            raise ValueError(
                f"Unknown solver_type {solver_type!r}; must be 'biconvex', "
                f"'polygonal', or 'nonconvex'."
            )
        self.trajectory_degree = trajectory_degree
        self.solver_type = solver_type
        self.rel_term = rel_term
        self.margin = margin
        self.split_margin = split_margin
        self.min_edge_length = min_edge_length

    def solve(
        self,
        path: np.ndarray,
        obstacles: List[pd.ConvexSet],
        domain: pd.HPolyhedron,
        limits: Limits,
        verbose: bool = False,
    ) -> SCSResult:
        """Plan a minimum-time trajectory through ``path`` avoiding ``obstacles``.

        ``path`` is an ``(N, dim)`` array of collision-free waypoints; the first
        and last are the start/goal. ``domain`` bounds the workspace. ``limits``
        must carry only velocity and acceleration -- the ``scsplanning`` backend
        has no jerk/snap support.
        """
        from scsplanning import nonconvex, polygonal

        if not isinstance(domain, pd.HPolyhedron):
            raise ValueError(f"domain must be an HPolyhedron, got {type(domain).__name__}.")
        if limits.order > 2:
            raise NotImplementedError(
                "EISCSPlanner supports only velocity and acceleration limits "
                f"(scsplanning has no jerk/snap); got Limits with order J={limits.order}. "
                "Use MinimumTimePlanner for jerk/snap-constrained planning."
            )
        velocity_set, acceleration_set = limits.velocity, limits.acceleration

        path = np.asarray(path, dtype=float)
        q_init, q_term = path[0], path[-1]
        t_start = time.time()

        t0 = time.time()
        regions = construct_regions_from_obstacles(path, obstacles, domain, margin=self.margin)
        region_time = time.time() - t0
        if verbose:
            print(f"[eiscs] built {len(regions)} region(s) in {region_time:.3f}s")

        t0 = time.time()
        solved_regions = regions
        if self.solver_type == "biconvex":
            trajectory, solved_regions = _biconvex_with_pruning(
                q_init,
                q_term,
                regions,
                velocity_set,
                acceleration_set,
                self.trajectory_degree,
                path,
                self.rel_term,
                self.split_margin,
                self.min_edge_length,
                verbose,
            )
        elif self.solver_type == "polygonal":
            trajectory = polygonal(
                q_init, q_term, regions, velocity_set, acceleration_set, self.trajectory_degree
            )
        else:  # nonconvex
            trajectory = nonconvex(
                q_init, q_term, regions, velocity_set, acceleration_set, self.trajectory_degree
            )
        trajopt_time = time.time() - t0
        trajectory = _as_float64_curve(trajectory)

        return SCSResult(
            trajectory=trajectory,
            total_time=trajectory.final_time - trajectory.initial_time,
            regions=solved_regions,
            num_regions=len(solved_regions),
            solver_type=self.solver_type,
            timing=dict(
                total=time.time() - t_start,
                region_generation=region_time,
                trajectory_optimization=trajopt_time,
            ),
        )


def solve_scs_minimum_time(
    path: np.ndarray,
    obstacles: List[pd.ConvexSet],
    domain: pd.HPolyhedron,
    limits: Limits,
    verbose: bool = False,
    **planner_kwargs,
) -> SCSResult:
    """One-shot convenience wrapper around :class:`EISCSPlanner`.

    Extra keyword arguments are forwarded to the :class:`EISCSPlanner`
    constructor (e.g. ``trajectory_degree``, ``solver_type``, ``rel_term``).
    """
    return EISCSPlanner(**planner_kwargs).solve(path, obstacles, domain, limits, verbose=verbose)
