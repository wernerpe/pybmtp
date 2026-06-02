"""Collision-free convex region construction for the SCS planner.

The SCS minimum-time formulation (Marcucci et al.) plans a Bezier trajectory
through a sequence of *overlapping convex regions* — one per path segment.
This module builds those regions directly from the obstacles by exact convex
separation, which is all the SCS+EI paper baseline ever needed (no GPU / EI-ZO):

  * :func:`construct_regions_from_obstacles` — for each path segment, intersect
    the domain with one separating half-space per obstacle. Because the input
    path is collision-free, every segment endpoint satisfies every half-space,
    so consecutive regions overlap at the shared waypoint and form a valid
    corridor by construction.
  * :func:`add_segment_splitting_hyperplanes` — enforce SCS *Assumption 1*
    (non-adjacent regions are disjoint) by cutting each ``(region_i,
    region_{i+2})`` pair apart with a plane through the middle segment's
    midpoint.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pydrake.all as pd

from ._solve import solve_in_parallel


def construct_regions_from_obstacles(
    path: np.ndarray,
    obstacles: List[pd.ConvexSet],
    domain: pd.HPolyhedron,
    margin: float = 0.0,
) -> List[pd.HPolyhedron]:
    """Build one collision-free convex region per path segment.

    For segment ``j`` (``path[j] -> path[j+1]``) we start from ``domain`` and,
    for every obstacle, add the separating half-space ``n.x <= n.p_obs - margin``
    where ``(p_obs, p_seg)`` are the closest points between the segment and the
    obstacle and ``n = (p_obs - p_seg) / ||.||``. All ``(segment, obstacle)``
    closest-point QPs are independent and solved together via
    :func:`pydrake.solvers.SolveInParallel`.

    Each QP packs ``x = [p_obs (dim), p_seg (dim), t (1)]`` and solves

        min  ||p_obs - p_seg||^2
        s.t. p_seg = path[j] + t (path[j+1] - path[j]),  t in [0, 1]
             p_obs in obstacle.

    The path is assumed collision-free, so every waypoint satisfies every
    half-space and consecutive regions overlap at their shared waypoint.

    Parameters
    ----------
    path:
        ``(N, dim)`` array of collision-free waypoints.
    obstacles:
        Convex obstacles (forbidden regions). May be empty.
    domain:
        Workspace domain.
    margin:
        Safety margin pushed toward the obstacle side.

    Returns
    -------
    List of ``N - 1`` HPolyhedra, one per segment.
    """
    num_segments = path.shape[0] - 1
    dim = path.shape[1]

    if not obstacles:
        return [pd.HPolyhedron(domain.A().copy(), domain.b().copy())
                for _ in range(num_segments)]

    # Quadratic cost ||p_obs - p_seg||^2 in x = [p_obs; p_seg; t]. Drake uses
    # the 1/2 x'Q x + b'x convention, so the coefficients are doubled.
    Q = np.zeros((2 * dim + 1, 2 * dim + 1))
    for i in range(dim):
        Q[i, i] = 2.0
        Q[dim + i, dim + i] = 2.0
        Q[i, dim + i] = -2.0
        Q[dim + i, i] = -2.0

    progs, x_list, seg_of_prog, obs_of_prog = [], [], [], []
    for j in range(num_segments):
        path_a, path_b = path[j], path[j + 1]
        d = path_b - path_a
        for obs_id, obs in enumerate(obstacles):
            prog = pd.MathematicalProgram()
            x = prog.NewContinuousVariables(2 * dim + 1, "x")

            # p_seg = path_a + t d  <=>  [I | -d] @ [p_seg; t] = path_a.
            A_eq = np.zeros((dim, 2 * dim + 1))
            A_eq[:, dim:2 * dim] = np.eye(dim)
            A_eq[:, 2 * dim] = -d
            prog.AddLinearEqualityConstraint(A_eq, path_a, x)

            prog.AddBoundingBoxConstraint(0.0, 1.0, np.array([x[2 * dim]]))
            obs.AddPointInSetConstraints(prog, x[:dim])
            prog.AddQuadraticCost(Q, np.zeros(2 * dim + 1), x, is_convex=True)

            progs.append(prog)
            x_list.append(x)
            seg_of_prog.append(j)
            obs_of_prog.append(obs_id)

    results = solve_in_parallel(progs)

    seg_A = [domain.A().copy() for _ in range(num_segments)]
    seg_b = [domain.b().copy() for _ in range(num_segments)]
    for result, j, x, obs_id in zip(results, seg_of_prog, x_list, obs_of_prog):
        if not result.is_success():
            raise RuntimeError(
                f"Region generation QP failed for segment {j}, obstacle "
                f"{obs_id}: {result.get_solution_result()}")

        x_sol = result.GetSolution(x)
        p_obs, p_seg = x_sol[:dim], x_sol[dim:2 * dim]
        diff = p_obs - p_seg
        dist = float(np.linalg.norm(diff))
        if dist <= 1e-6:
            # Segment touches the obstacle: the normal is unreliable and the
            # half-space would cut off the path endpoints. Skip it; the
            # trajectory through this region may then graze the obstacle.
            continue

        n = diff / dist
        # If the requested margin exceeds the available clearance, relax it so
        # the half-space still contains both segment endpoints.
        relaxation = max(0.0, margin - 0.999 * dist)
        c = float(n @ p_obs) - margin + relaxation
        seg_A[j] = np.vstack([seg_A[j], n.reshape(1, -1)])
        seg_b[j] = np.concatenate([seg_b[j], [c]])

    return [pd.HPolyhedron(A, b) for A, b in zip(seg_A, seg_b)]


def add_segment_splitting_hyperplanes(
    path: np.ndarray,
    regions: List[pd.HPolyhedron],
    original_indices: Optional[List[int]] = None,
    split_margin: float = 1e-4,
) -> List[pd.HPolyhedron]:
    """Enforce SCS *Assumption 1* (non-adjacent regions disjoint).

    For each triple ``(region_i, region_{i+1}, region_{i+2})``, let the middle
    segment be ``wp_a -> wp_b`` (the segment ``region_{i+1}`` covers). With unit
    normal ``n = (wp_b - wp_a) / ||.||`` and midpoint ``m = (wp_a + wp_b) / 2``,
    the splitting plane ``n.x = c`` (``c = n.m``) separates them. We add

        regions[i]    :   n.x <=  c - split_margin   (keep the wp_a side)
        regions[i+2]  :  -n.x <= -(c + split_margin) (keep the wp_b side)

    so the strip ``(c - margin, c + margin)`` belongs to neither region and
    ``IntersectsWith`` returns False even for closed polyhedra.

    Safe for any forward-progressing path (``n.wp[i] <= n.wp[i+1]``). A
    backtrack would cut a waypoint out of its own region; an endpoint-containment
    check at the end raises loudly if that happens.

    ``original_indices[k]`` maps surviving region ``k`` back to its segment index
    in ``path`` (identity if no pruning was done).
    """
    if original_indices is None:
        original_indices = list(range(len(regions)))

    if len(regions) < 3:
        return list(regions)

    new_regions = list(regions)
    for i in range(len(regions) - 2):
        mid_seg_idx = original_indices[i + 1]
        wp_a, wp_b = path[mid_seg_idx], path[mid_seg_idx + 1]
        d = wp_b - wp_a
        d_norm = float(np.linalg.norm(d))
        if d_norm < 1e-9:
            continue
        n = d / d_norm
        c = 0.5 * (float(n @ wp_a) + float(n @ wp_b))

        A_i, b_i = new_regions[i].A(), new_regions[i].b()
        new_regions[i] = pd.HPolyhedron(
            np.vstack([A_i, n.reshape(1, -1)]),
            np.concatenate([b_i, [c - split_margin]]))

        A_i2, b_i2 = new_regions[i + 2].A(), new_regions[i + 2].b()
        new_regions[i + 2] = pd.HPolyhedron(
            np.vstack([A_i2, (-n).reshape(1, -1)]),
            np.concatenate([b_i2, [-(c + split_margin)]]))

    tol = split_margin
    for k, region in enumerate(new_regions):
        seg_idx = original_indices[k]
        for label, wp in (("start", path[seg_idx]), ("end", path[seg_idx + 1])):
            residual = float((region.A() @ wp - region.b()).max())
            if residual > tol:
                raise AssertionError(
                    f"add_segment_splitting_hyperplanes: region {k} no longer "
                    f"contains its {label} waypoint (orig segment {seg_idx}, "
                    f"residual={residual:+.3e} > tol={tol:.0e}). Path likely "
                    f"backtracks past its successor's midpoint.")

    return new_regions
