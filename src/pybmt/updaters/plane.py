"""Plane-update LP for the biconvex minimum-time planner.

With the trajectory fixed, this finds, for each active ``(segment, obstacle)``
pair, a degree-``plane_degree`` Bezier plane ``(a(t), b(t))`` that

  * supports the obstacle (``a(t).x + b(t) >= 0`` for all ``x`` in the obstacle,
    enforced via conic duality in :func:`pybmt.geometry.add_supporting_plane`),
  * and separates the trajectory as strongly as possible: minimize the worst-
    case margin ``max_t (a(t).r(t) + b(t))`` over the segment.

A negative optimal margin means the plane strictly separates the trajectory
from the obstacle; that plane then becomes a half-space constraint in the next
trajectory-update solve. Sub-problems are independent and solved in parallel.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pybezier as pb
import pydrake.all as pd
from scipy.special import comb as _comb

from .._solve import solve_in_parallel
from ..geometry import add_supporting_plane


def _build_plane_objective_matrix(
    r_points: np.ndarray,
    plane_degree: int,
    traj_degree: int,
    dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numeric matrix form of ``a(t).r(t) + b(t) <= margin`` for all t.

    Here the *trajectory* ``r`` is fixed (numeric) and the plane ``(a, b)`` plus
    ``margin`` are the decision variables, so the constraint is linear in them.

    Readable symbolic equivalent (the form this expands; uses
    :func:`pybmt.bezier_ops.bezier_dot_product` with symbolic ``a``/``b``)::

        prod = bezier_dot_product(a_curve, r_curve) + b_curve   # degree m+n
        prog.AddConstraint(pd.le(prod.points.flatten() - margin, 0))

    By the Bernstein convex-hull property, bounding the ``m+n+1`` product control
    points bounds the curve everywhere. Each product control point is

        c_i = sum_j [C(m,j) C(n,k)/C(m+n,i)] * (a_j . r_k) + b_elev_i,  k = i-j

    which is linear in the flattened plane variables. We assemble

        [ A_a | A_b | -1 ] @ [a_flat; b_flat; margin] <= 0

    where ``A_a`` carries the ``r_k``-weighted normal coefficients and ``A_b`` is
    the degree-elevation of ``b`` from degree ``m`` up to ``m+n``.

    Returns ``(A_full, lb, ub)`` with ``lb = -inf``, ``ub = 0``.
    """
    m, n = plane_degree, traj_degree
    prod_d = m + n
    n_out = prod_d + 1

    A_a = np.zeros((n_out, (m + 1) * dim))
    for i in range(n_out):
        for j in range(max(0, i - n), min(m, i) + 1):
            k = i - j
            coeff = float(_comb(m, j, exact=True) * _comb(n, k, exact=True)
                          / _comb(prod_d, i, exact=True))
            A_a[i, j * dim:(j + 1) * dim] = coeff * r_points[k]

    elev_d = n
    A_b = np.zeros((n_out, m + 1))
    for i in range(n_out):
        for j in range(max(0, i - elev_d), min(m, i) + 1):
            A_b[i, j] = float(_comb(m, j, exact=True) * _comb(elev_d, i - j, exact=True)
                              / _comb(prod_d, i, exact=True))

    A_full = np.hstack([A_a, A_b, -np.ones((n_out, 1))])
    return A_full, np.full(n_out, -np.inf), np.zeros(n_out)


class PlaneUpdater:
    """Builds and solves the separating-plane LPs for a fixed trajectory.

    Parameters
    ----------
    obstacles:
        List of convex obstacles (HPolyhedron / Hyperellipsoid / Hyperrectangle
        / VPolytope). Indexed by the obstacle ids used in ``tagged_obstacles``.
    plane_degree:
        Bezier degree of each separating plane (1 is usually sufficient).
    plane_to_obstacle_tol:
        ``step_back`` used in the supporting-plane constraint — how strictly the
        plane must clear the obstacle surface.
    """

    def __init__(
        self,
        obstacles: List[pd.ConvexSet],
        plane_degree: int = 1,
        plane_to_obstacle_tol: float = 1e-9,
    ):
        self.obstacles = obstacles
        # dim is only needed once a (segment, obstacle) pair is processed; with
        # no obstacles the updater is a no-op and never touches it.
        self.dim = obstacles[0].ambient_dimension() if obstacles else None
        self.plane_degree = plane_degree
        self.plane_to_obstacle_tol = plane_to_obstacle_tol
        self.solver = pd.ClarabelSolver()
        self._solver_opts = pd.SolverOptions()
        self._solver_opts.SetOption(pd.CommonSolverOption.kPrintToConsole, False)
        # Lazy per-obstacle base programs: obs_id -> (prog, a_flat, b_flat).
        self._base_progs: dict = {}

    def _get_base_prog(self, obs_id: int) -> Tuple[pd.MathematicalProgram, np.ndarray, np.ndarray]:
        """Lazily build (and cache) the fixed obstacle-structure program.

        The base program holds only the supporting-plane (conic dual) constraints
        for the obstacle's ``plane_degree + 1`` control points. It carries no
        trajectory-dependent terms, so it can be ``Clone()``-d and have the
        margin objective added for each segment that needs this obstacle.
        """
        if obs_id in self._base_progs:
            return self._base_progs[obs_id]

        obs = self.obstacles[obs_id]
        if isinstance(obs, pd.Hyperrectangle):
            obs = obs.MakeHPolyhedron()

        prog = pd.MathematicalProgram()
        a_cps, b_cps = [], []
        for cp_id in range(self.plane_degree + 1):
            a_var, b_var = add_supporting_plane(
                prog, obs, f"o{obs_id}_cp{cp_id}",
                norm_val=1, restrict_norm=True, step_back=self.plane_to_obstacle_tol)
            a_cps.append(a_var)
            b_cps.append(b_var.flatten())

        a_flat = np.concatenate(a_cps)
        b_flat = np.concatenate(b_cps)
        self._base_progs[obs_id] = (prog, a_flat, b_flat)
        return self._base_progs[obs_id]

    def compute_planes(
        self,
        tagged_obstacles: Dict[int, List[int]],
        trajectory: pb.CompositeBezierCurve,
    ) -> Dict[Tuple[int, int], Tuple[pb.BezierCurve, pb.BezierCurve]]:
        """Solve the separating-plane LP for every tagged ``(segment, obstacle)``.

        ``tagged_obstacles`` maps a segment index to the obstacle indices that
        segment must be separated from (typically the output of the collision
        check). Returns ``{(segment_id, obstacle_id): (a_curve, b_curve)}``.
        """
        segments = trajectory.curves
        pairs = [
            (seg_id, obs_id)
            for seg_id, obs_ids in tagged_obstacles.items()
            for obs_id in obs_ids
        ]
        if not pairs:
            return {}

        clones = []
        meta = []
        for seg_id, obs_id in pairs:
            base_prog, a_flat, b_flat = self._get_base_prog(obs_id)
            clone = base_prog.Clone()

            r_pts = np.asarray(segments[seg_id].points, dtype=float)
            traj_degree = len(r_pts) - 1
            A_full, lb, ub = _build_plane_objective_matrix(
                r_pts, self.plane_degree, traj_degree, self.dim)

            margin = clone.NewContinuousVariables(1, "margin")
            clone.AddLinearConstraint(A_full, lb, ub, np.concatenate([a_flat, b_flat, margin]))
            clone.AddLinearCost(margin[0])

            clones.append(clone)
            meta.append((seg_id, obs_id, a_flat, b_flat))

        results = solve_in_parallel(clones, self._solver_opts)

        planes: Dict[Tuple[int, int], Tuple[pb.BezierCurve, pb.BezierCurve]] = {}
        for result, (seg_id, obs_id, a_flat, b_flat) in zip(results, meta):
            if not result.is_success():
                raise RuntimeError(
                    f"Plane LP failed for seg={seg_id} obs={obs_id}: "
                    f"{result.get_solution_result()}")
            # A negative optimal margin means the plane strictly separates the
            # (feasible) trajectory from the obstacle. A clearly positive margin
            # means no separating plane exists at this degree — usually because
            # the trajectory passed in is not actually collision-free. The small
            # tolerance absorbs solver noise around an exactly-touching margin.
            if result.get_optimal_cost() > 1e-6:
                raise RuntimeError(
                    f"No separating plane for seg={seg_id} obs={obs_id} "
                    f"(best margin={result.get_optimal_cost():.3e} > 0). The "
                    f"trajectory may not be collision-free, or try a higher "
                    f"plane_degree.")
            r = segments[seg_id]
            a_sol = result.GetSolution(a_flat).reshape(self.plane_degree + 1, self.dim)
            b_sol = result.GetSolution(b_flat).reshape(-1, 1)
            planes[(seg_id, obs_id)] = (
                pb.BezierCurve(a_sol, r.initial_time, r.final_time),
                pb.BezierCurve(b_sol, r.initial_time, r.final_time),
            )
        return planes
