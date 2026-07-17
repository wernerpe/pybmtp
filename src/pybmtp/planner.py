"""The biconvex minimum-time planner (BMTP) and its one-shot wrapper.

The planner alternates two convex sub-problems until the trajectory stops
getting faster:

  1. **Trajectory update** (:class:`~pybmtp.updaters.TrajectoryUpdater`): with the
     current separating planes fixed, solve a SOCP for the fastest trajectory
     that satisfies the velocity/acceleration/continuity constraints and stays
     on the safe side of every plane.
  2. **Collision check** (:mod:`pybmtp.collisions`): test the candidate against
     the obstacles. Segments that are not provably clear get *tagged*.
  3. **Plane update** (:class:`~pybmtp.updaters.PlaneUpdater`): for each tagged
     ``(segment, obstacle)`` pair, compute a separating plane that keeps the
     last *feasible* trajectory safe, and feed it back into step 1.

Once a candidate is collision-free, it becomes the new feasible iterate; the
loop terminates when the relative segment-time improvement between consecutive
feasible iterates drops below ``rel_term``.

Obstacles must be ``HPolyhedron`` instances: the collision check uses the
Bezier-vs-HPolyhedron kernel, and the plane update supports them directly.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pybezier as pb
import pydrake.all as pd

from .collisions import intersect_with_hpolyhedra
from .limits import Limits
from .polygonal import polygonal_initialization
from .result import SolveResult
from .updaters import PlaneUpdater, TrajectoryUpdater


def _merge_tagged(
    cumulative: Dict[int, List[int]], new: Dict[int, List[int]]
) -> Dict[int, List[int]]:
    """Union ``new`` collision tags into the cumulative tag map."""
    merged = {k: list(v) for k, v in cumulative.items()}
    for seg, obs in new.items():
        merged[seg] = sorted(set(merged.get(seg, [])) | set(obs))
    return merged


class MinimumTimePlanner:
    """Biconvex minimum-time Bezier trajectory planner.

    Configuration is set once at construction; :meth:`solve` runs the biconvex
    loop for a specific problem instance.

    Parameters
    ----------
    trajectory_degree:
        Bezier degree of each trajectory segment (>= 5 so zero boundary
        velocity and acceleration can both be enforced).
    plane_degree:
        Bezier degree of each separating plane.
    rel_term:
        Termination threshold on relative per-segment-time improvement between
        consecutive feasible iterates.
    max_iter:
        Maximum biconvex outer iterations.
    continuity_order:
        Enforce ``C1 ... C_{continuity_order}`` continuity at segment junctions.
        ``None`` (default) uses the limits' order ``J`` (e.g. C1+C2 for velocity
        + acceleration limits). May not exceed ``J``.
    terminal_order:
        Pin derivatives ``1 ... terminal_order`` to zero at the start and end.
        ``None`` (default) uses ``J`` (e.g. zero terminal velocity + acceleration).
    trajectory_to_plane_tol:
        Positive buffer pushing the trajectory off each plane. (The underlying
        LP enforces ``a.r + b <= tol``; the sign is flipped internally so a
        positive value here is a true safety buffer.)
    plane_to_obstacle_tol:
        Buffer keeping each supporting plane clear of the obstacle surface.
    collision_check_tol:
        Tolerance of the Bezier collision kernel (subdivision resolution).
    """

    def __init__(
        self,
        trajectory_degree: int = 6,
        plane_degree: int = 1,
        rel_term: float = 0.05,
        max_iter: int = 100,
        continuity_order: int | None = None,
        terminal_order: int | None = None,
        trajectory_to_plane_tol: float = 1e-8,
        plane_to_obstacle_tol: float = 1e-9,
        collision_check_tol: float = 0.05,
    ):
        assert trajectory_degree >= 5, (
            "trajectory_degree must be >= 5 to satisfy zero boundary velocity "
            "and acceleration at both ends"
        )
        self.trajectory_degree = trajectory_degree
        self.plane_degree = plane_degree
        self.rel_term = rel_term
        self.max_iter = max_iter
        self.continuity_order = continuity_order
        self.terminal_order = terminal_order
        self.trajectory_to_plane_tol = trajectory_to_plane_tol
        self.plane_to_obstacle_tol = plane_to_obstacle_tol
        self.collision_check_tol = collision_check_tol

    def solve(
        self,
        path: np.ndarray,
        obstacles: List[pd.HPolyhedron],
        domain: pd.HPolyhedron,
        limits: Limits,
        verbose: bool = False,
    ) -> SolveResult:
        """Plan a minimum-time trajectory through ``path`` avoiding ``obstacles``.

        ``path`` is an ``(N, dim)`` array of waypoints; the first and last are
        the start/goal. ``domain`` bounds the workspace. ``limits`` bundles the
        velocity / acceleration (and optional jerk / snap) sets.
        """
        path = np.asarray(path, dtype=float)
        t_start = time.time()

        # --- warm start: constraint-feasible polygonal trajectory ------------
        init = polygonal_initialization(
            path,
            domain,
            limits,
            degree=self.trajectory_degree,
            continuity_order=self.continuity_order,
            terminal_order=self.terminal_order,
            force_same_segment_time=True,
        )
        num_segments = len(init.curves)
        segment_time_init = init.curves[0].duration
        # Re-time onto the normalized [seg/N, (seg+1)/N] grid the updater uses.
        r_feasible = _retime_normalized(init, num_segments)

        traj_updater = TrajectoryUpdater(
            path[0],
            path[-1],
            domain,
            num_segments,
            limits,
            degree=self.trajectory_degree,
            continuity_order=self.continuity_order,
            terminal_order=self.terminal_order,
        )
        plane_updater = PlaneUpdater(
            obstacles, self.plane_degree, plane_to_obstacle_tol=self.plane_to_obstacle_tol
        )

        new_planes: Dict[Tuple[int, int], Tuple[pb.BezierCurve, pb.BezierCurve]] = {}
        all_planes: Dict[Tuple[int, int], Tuple[pb.BezierCurve, pb.BezierCurve]] = {}
        tagged: Dict[int, List[int]] = {}
        last_segment_time = segment_time_init

        feasible_trajectories = [r_feasible]
        feasible_segment_times = [segment_time_init]
        feasible_wall_times = [0.0]

        timing = dict(
            trajectory_update=0.0, trajectory_solver=0.0, collision_check=0.0, plane_update=0.0
        )
        converged = False
        iterations = 0

        for _ in range(self.max_iter):
            iterations += 1

            # 1. Trajectory update. The LP enforces a.r + b <= tol; pass the
            #    negated buffer so a positive trajectory_to_plane_tol pushes the
            #    trajectory away from the plane.
            t0 = time.time()
            r_cand, segment_time, _, solver_time = traj_updater.solve(
                new_planes, tol=-self.trajectory_to_plane_tol
            )
            timing["trajectory_update"] += time.time() - t0
            timing["trajectory_solver"] += solver_time

            # 2. Collision check (ignore already-tagged pairs — planes cover them).
            t0 = time.time()
            new_tagged = intersect_with_hpolyhedra(
                r_cand, obstacles, ignore=tagged, tol=self.collision_check_tol
            )
            tagged = _merge_tagged(tagged, new_tagged)
            timing["collision_check"] += time.time() - t0

            # 3. Plane update.
            t0 = time.time()
            if not new_tagged:
                # Candidate is collision-free: accept it as the feasible iterate.
                r_feasible = r_cand
                traj_updater.clear_planes()
                rel_improvement = (last_segment_time - segment_time) / last_segment_time
                last_segment_time = segment_time
                feasible_segment_times.append(segment_time)
                feasible_trajectories.append(r_cand)
                feasible_wall_times.append(time.time() - t_start)
                if verbose:
                    print(
                        f"[pybmtp] feasible: T={segment_time * num_segments:.3f}s "
                        f"rel_improvement={rel_improvement:.4f}"
                    )
                if rel_improvement < self.rel_term:
                    converged = True
                    timing["plane_update"] += time.time() - t0
                    break
                # Recompute all planes against the new feasible trajectory.
                new_planes = plane_updater.compute_planes(tagged, r_feasible)
                all_planes = dict(new_planes)
            else:
                if verbose:
                    print(f"[pybmtp] {len(new_tagged)}/{num_segments} segments in collision")
                new_planes = plane_updater.compute_planes(new_tagged, r_feasible)
                all_planes.update(new_planes)
            timing["plane_update"] += time.time() - t0

        # Time-scale the final feasible trajectory into physical seconds.
        r_sol = feasible_trajectories[-1]
        T_sol = feasible_segment_times[-1]
        trajectory = pb.CompositeBezierCurve(
            [
                pb.BezierCurve(s.points, i * T_sol, (i + 1) * T_sol)
                for i, s in enumerate(r_sol.curves)
            ]
        )
        timing["total"] = time.time() - t_start

        return SolveResult(
            trajectory=trajectory,
            segment_time=T_sol,
            total_time=T_sol * num_segments,
            num_iterations=iterations,
            converged=converged,
            planes=all_planes,
            feasible_trajectories=feasible_trajectories,
            feasible_segment_times=feasible_segment_times,
            feasible_wall_times=feasible_wall_times,
            timing=timing,
        )


def _retime_normalized(
    curve: pb.CompositeBezierCurve, num_segments: int
) -> pb.CompositeBezierCurve:
    """Re-time a composite curve onto the uniform [seg/N, (seg+1)/N] grid."""
    return pb.CompositeBezierCurve(
        [
            pb.BezierCurve(c.points, i / num_segments, (i + 1) / num_segments)
            for i, c in enumerate(curve.curves)
        ]
    )


def solve_minimum_time(
    path: np.ndarray,
    obstacles: List[pd.HPolyhedron],
    domain: pd.HPolyhedron,
    limits: Limits,
    verbose: bool = False,
    **planner_kwargs,
) -> SolveResult:
    """One-shot convenience wrapper around :class:`MinimumTimePlanner`.

    Extra keyword arguments are forwarded to the :class:`MinimumTimePlanner`
    constructor (e.g. ``trajectory_degree``, ``rel_term``, ``max_iter``).
    """
    return MinimumTimePlanner(**planner_kwargs).solve(
        path, obstacles, domain, limits, verbose=verbose
    )
