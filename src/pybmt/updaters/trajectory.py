"""Trajectory-update SOCP for the biconvex minimum-time planner.

Given a fixed set of separating planes (one per active trajectory-segment /
obstacle pair), this solves for the Bezier control points and the common
segment duration ``T`` that minimize total time while satisfying:

  * boundary conditions (position, and optionally zero derivatives at the ends),
  * per-derivative limits up to order ``J`` (convex-set membership of the Bezier
    derivative control points, scaled by powers of ``T``),
  * ``C1 ... C_{continuity_order}`` continuity between segments,
  * the half-space constraints from the current planes.

All segments share one duration ``T`` (the time grid is uniform). The highest
constrained derivative order ``J`` (2 = acceleration, 3 = jerk, 4 = snap) is read
from the :class:`~pybmt.limits.Limits` object, and the rotated-Lorentz "power
ladder" ``T_powers = [1, T, T**2, ..., T**J]`` is built *only* that deep -- the
k-th derivative limit scales with ``T_powers[k]``. With only velocity and
acceleration limits this is exactly the ``[1, T, T**2]`` program; adding jerk /
snap grows the ladder to ``T**3`` / ``T**4`` and nothing else changes.

Two equivalent constraint encodings live here:

  * a **symbolic** form built from ``pydrake`` expressions and Bezier
    derivative curves -- readable, and the reference implementation;
  * a **fast matrix** form that writes the same affine constraints directly as
    ``AddLinearConstraint(A, lb, ub, vars)``, skipping Drake's symbolic-formula
    parsing (which dominated build time).

Both share one general finite-difference formula for the k-th Bezier
derivative, so they reduce to identical velocity/acceleration constraints at
``J = 2``. The ``use_symbolic_constraints`` flag forces the symbolic path even
for HPolyhedron sets, which the test-suite uses to assert the two forms produce
identical solutions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pybezier as pb
import pydrake.all as pd
from scipy.special import comb as _comb

from ..bezier_ops import bezier_dot_product
from ..limits import Limits


def _fd_coeffs(k: int) -> np.ndarray:
    """Coefficients of the k-th forward finite difference: ``(-1)^(k-j) C(k, j)``.

    ``Delta^k p[i] = sum_{j=0}^{k} (-1)^(k-j) C(k,j) p[i+j]``.
    """
    return np.array([(-1) ** (k - j) * _comb(k, j, exact=True) for j in range(k + 1)], float)


def _falling_factorial(d: int, k: int) -> float:
    """``d! / (d-k)! = d (d-1) ... (d-k+1)`` -- the Bezier k-th-derivative scale."""
    r = 1.0
    for i in range(k):
        r *= d - i
    return r


def _plane_constraint_matrix(
    a_points: np.ndarray,
    b_points: np.ndarray,
    traj_degree: int,
    plane_degree: int,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Numeric matrix form of the half-space constraint ``a(t).r(t)+b(t) <= tol``.

    Readable symbolic equivalent (what this expands; see
    :func:`pybmt.bezier_ops.bezier_dot_product`)::

        prod = bezier_dot_product(a_curve, r_curve) + b_curve   # degree m+n
        prog.AddConstraint(pd.le(prod.points.flatten(), tol))   # convex hull

    Here ``r``'s control points are the decision variables, so the product's
    control points are affine in the flattened ``r`` vector. We assemble that
    affine map explicitly:

        c_i = sum_j [C(m,j) C(n,k) / C(m+n,i)] * (a_j . r_k),   k = i - j

    so the coefficient multiplying ``r_k`` (a ``dim``-vector) in row i is
    ``[C(m,j) C(n,k)/C(m+n,i)] * a_j``. The offset curve ``b`` is degree-
    elevated from ``m`` to ``m+n`` and moved to the right-hand side.

    Returns ``(A_mat, ub_vec)`` for ``A_mat @ r_flat <= ub_vec`` where
    ``r_flat`` is the segment's control points flattened row-major (degree+1, dim).
    """
    dim = a_points.shape[1]
    m = plane_degree
    n = traj_degree
    prod_d = m + n
    n_out = prod_d + 1

    # Affine map from flattened r control points to the product control points.
    A_mat = np.zeros((n_out, (n + 1) * dim))
    for i in range(n_out):
        for j in range(max(0, i - n), min(m, i) + 1):
            k = i - j
            coeff = float(
                _comb(m, j, exact=True) * _comb(n, k, exact=True) / _comb(prod_d, i, exact=True)
            )
            A_mat[i, k * dim : (k + 1) * dim] += coeff * np.asarray(a_points[j], float)

    # Degree-elevate b from degree m up to prod_d, then ub = tol - b_elevated.
    elev_d = prod_d - m  # == n
    b_elev = np.zeros(n_out)
    for i in range(n_out):
        for j in range(max(0, i - elev_d), min(m, i) + 1):
            coeff = float(
                _comb(m, j, exact=True)
                * _comb(elev_d, i - j, exact=True)
                / _comb(prod_d, i, exact=True)
            )
            b_elev[i] += coeff * float(b_points[j])

    return A_mat, np.full(n_out, tol) - b_elev


class TrajectoryUpdater:
    """Builds and repeatedly solves the trajectory-update SOCP.

    Construction wires up the time-invariant constraints (boundary, derivative
    limits, continuity); planes are added per :meth:`solve` call and can be
    removed with :meth:`clear_planes`, so the same program object is reused
    across biconvex iterations.

    Parameters
    ----------
    source, target:
        Start/goal points in task space (length ``dim``).
    domain:
        HPolyhedron the trajectory must stay inside (applied to interior control
        points via the Bernstein convex-hull property).
    num_segments:
        Number of Bezier segments.
    limits:
        :class:`~pybmt.limits.Limits` bundling velocity + acceleration and,
        optionally, jerk and snap sets. Its :attr:`~pybmt.limits.Limits.order`
        ``J`` sets the depth of the ``T_powers`` ladder.
    degree:
        Bezier degree of each trajectory segment. Must satisfy
        ``degree >= 2 * terminal_order + 1`` (so terminal-rest control points do
        not overlap) and ``degree >= continuity_order + 1``.
    continuity_order:
        Enforce ``C1 ... C_{continuity_order}`` continuity at segment junctions.
        Defaults to ``J``. May not exceed ``J``: constraining the continuity of a
        derivative that is itself unconstrained would fight the minimum-time
        objective (that derivative is driven to infinity), so this raises.
    terminal_order:
        Pin derivatives ``1 ... terminal_order`` to zero at the start and end
        (rest-to-rest). Defaults to ``J`` (e.g. an acceleration limit gives
        ``r''(0) = r''(1) = 0``). ``0`` leaves the endpoints free apart from
        position. May not exceed ``J``.
    use_symbolic_constraints:
        Force the readable symbolic constraint path even for HPolyhedron sets.
        Default False (fast matrix path). Mainly for validation/testing.
    """

    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        domain: pd.HPolyhedron,
        num_segments: int,
        limits: Limits,
        degree: int = 6,
        continuity_order: Optional[int] = None,
        terminal_order: Optional[int] = None,
        use_symbolic_constraints: bool = False,
    ):
        self.source = np.asarray(source, float)
        self.target = np.asarray(target, float)
        self.dim = len(self.source)
        self.N = num_segments
        self.degree = degree
        self.domain = domain

        self.limits = limits
        self.deriv_sets = limits.sets  # [velocity, acceleration, (jerk), (snap)]
        self.J = limits.order

        self.continuity_order = self.J if continuity_order is None else int(continuity_order)
        self.terminal_order = self.J if terminal_order is None else int(terminal_order)

        if self.continuity_order > self.J:
            raise ValueError(
                f"continuity_order ({self.continuity_order}) cannot exceed the highest "
                f"constrained derivative order J={self.J}: the C{self.continuity_order} "
                "constraint would couple an unconstrained derivative, which the minimum-time "
                "objective drives to infinity."
            )
        if not 0 <= self.terminal_order <= self.J:
            raise ValueError(
                f"terminal_order ({self.terminal_order}) must be between 0 and J={self.J}."
            )
        if degree < 2 * self.terminal_order + 1:
            raise ValueError(
                f"degree ({degree}) must be >= 2*terminal_order+1 = {2 * self.terminal_order + 1} "
                "so the zero-derivative control points at the two ends do not overlap."
            )
        if degree < self.continuity_order + 1:
            raise ValueError(
                f"degree ({degree}) must be >= continuity_order+1 = {self.continuity_order + 1}."
            )

        hpoly_sets = all(isinstance(s, pd.HPolyhedron) for s in self.deriv_sets)
        self.use_symbolic_constraints = use_symbolic_constraints or not hpoly_sets

        self.solver = pd.ClarabelSolver()
        self.prog = pd.MathematicalProgram()
        self._solver_opts = pd.SolverOptions()
        self._solver_opts.SetOption(pd.CommonSolverOption.kPrintToConsole, False)
        self.added_plane_constraints: list = []

        self._build_time_variables()
        self._build_segments()
        self._build_boundary_constraints()
        if self.use_symbolic_constraints:
            self._build_limits_and_continuity_symbolic()
        else:
            self._build_limits_and_continuity_matrix()
        self._build_terminal_rest_constraints()

    # ------------------------------------------------------------------ setup

    def _build_time_variables(self) -> None:
        # T_powers = [1, T, T**2, ..., T**J]. The rotated Lorentz cone
        # T_powers[j-2]*T_powers[j] >= T_powers[j-1]**2 (with T_powers[0] == 1)
        # forces T_powers[j] >= T**j while staying convex. The k-th derivative
        # limit scales with T_powers[k]; we build the ladder exactly J deep.
        self.T_powers = self.prog.NewContinuousVariables(self.J + 1, "T_powers")
        self.prog.AddLinearEqualityConstraint(self.T_powers[0] == 1)
        for j in range(2, self.J + 1):
            A = np.zeros((3, self.J + 1))
            A[0, j - 2] = 1
            A[1, j] = 1
            A[2, j - 1] = 1
            self.prog.AddRotatedLorentzConeConstraint(A=A, b=np.zeros(3), vars=self.T_powers)
        self.prog.AddLinearCost(self.T_powers[-1])  # minimize T**J (monotone in T)

    def _build_segments(self) -> None:
        # Control points are decision variables, except shared junction points
        # (C0 continuity: a segment's first point IS the previous segment's
        # last point) and the boundary-locked endpoint groups.
        self.untimed_segments: List[pb.BezierCurve] = []
        n_cp = self.degree + 1
        n_pin = self.terminal_order + 1  # coincident control points pinned at each end
        prev_last = None
        for seg_id in range(self.N):
            cps = []
            for i in range(n_cp):
                if prev_last is not None and i == 0:
                    cp = prev_last  # C0: reuse previous segment's last control point
                else:
                    cp = self.prog.NewContinuousVariables(self.dim, f"seg{seg_id}_cp{i}")
                cps.append(cp)
                if i == n_cp - 1:
                    prev_last = cp

                # The first/last n_pin control points are pinned onto source/target
                # by the boundary + zero-terminal-derivative constraints, so they
                # are already inside the domain and skip the domain box.
                if seg_id == 0 and i < n_pin:
                    continue
                if seg_id == self.N - 1 and i >= n_cp - n_pin:
                    continue
                # i == 0 is a shared junction; its domain membership is handled
                # by the owning (previous) segment's last point.
                if i == 0:
                    continue
                self.domain.AddPointInSetConstraints(self.prog, cp)
            self.untimed_segments.append(
                pb.BezierCurve(np.array(cps), initial_time=0, final_time=1)
            )

        self._seg_vars_flat = [seg.points.flatten() for seg in self.untimed_segments]

    def _build_boundary_constraints(self) -> None:
        c = self.prog.AddLinearConstraint(pd.eq(self.untimed_segments[0].points[0], self.source))
        c.evaluator().set_description("boundary_initial_position")
        c = self.prog.AddLinearConstraint(pd.eq(self.untimed_segments[-1].points[-1], self.target))
        c.evaluator().set_description("boundary_terminal_position")

    # ---------------------------------------------- limits + continuity (fast)

    def _build_limits_and_continuity_matrix(self) -> None:
        """Fast numeric matrix encoding of the derivative limits and continuity.

        For every derivative order ``k = 1 ... J`` the k-th Bezier derivative
        control point is a finite difference of ``k+1`` position control points,
        scaled by the falling factorial (see :func:`_fd_coeffs`)::

            s^(k)[i] = ff(d,k) * sum_{j=0}^{k} fd_k[j] * pts[i+j]

        and its limit is ``set_A @ s^(k)[i] <= set_b * T_powers[k]``. At ``k = 1``
        and ``k = 2`` this reproduces exactly the hand-written velocity /
        acceleration blocks.
        """
        d = self.degree
        n_segs = len(self.untimed_segments)
        eye = np.eye(self.dim)

        # Precompute the per-order limit constraint row-block [ff*fd_k[0]*A | ... | ff*fd_k[k]*A | -b].
        limit_blocks = []
        for k in range(1, self.J + 1):
            A_set, b_set = self.deriv_sets[k - 1].A(), self.deriv_sets[k - 1].b()
            fd = _fd_coeffs(k)
            ff = _falling_factorial(d, k)
            A_row = np.hstack([ff * fd[j] * A_set for j in range(k + 1)] + [-b_set.reshape(-1, 1)])
            lb = np.full(A_set.shape[0], -np.inf)
            ub = np.zeros(A_set.shape[0])
            limit_blocks.append((A_row, lb, ub))

        for s_idx, s in enumerate(self.untimed_segments):
            pts = s.points
            for k in range(1, self.J + 1):
                A_row, lb, ub = limit_blocks[k - 1]
                start, stop = self._derivative_index_range(s_idx, k, n_segs)
                for i in range(start, stop):
                    vs = np.concatenate([pts[i + j] for j in range(k + 1)] + [[self.T_powers[k]]])
                    c = self.prog.AddLinearConstraint(A_row, lb, ub, vs)
                    c.evaluator().set_description(f"deriv{k}_seg{s_idx}_cp{i}")

        # The continuity coefficient pattern depends only on (degree, k), not on
        # which junction, so build each row's matrix + control-point selector once
        # and reuse it across all N-1 junctions.
        cont_patterns = {
            k: self._continuity_pattern(k, eye)
            for k in range(1, self.continuity_order + 1)
            if self.degree >= k + 1
        }
        for s_id in range(n_segs - 1):
            s1, s2 = self.untimed_segments[s_id], self.untimed_segments[s_id + 1]
            pts = {"s1": s1.points, "s2": s2.points}
            for k, (A_c, keys) in cont_patterns.items():
                vars_c = np.concatenate([pts[side][idx] for side, idx in keys])
                c = self.prog.AddLinearConstraint(
                    A_c, np.zeros(self.dim), np.zeros(self.dim), vars_c
                )
                c.evaluator().set_description(f"C{k}_seg{s_id}_to_{s_id + 1}")

    def _continuity_pattern(self, k: int, eye: np.ndarray) -> Tuple[np.ndarray, list]:
        """Reusable matrix form of ``s1^(k)[-1] == s2^(k)[0]`` (the falling factor cancels).

        ``s1^(k)[-1]`` uses ``s1.pts[d-k .. d]`` and ``s2^(k)[0]`` uses
        ``s2.pts[0 .. k]``; the shared junction point (``s1.pts[-1]`` is
        ``s2.pts[0]`` under C0) has its coefficients merged, so this reduces to the
        familiar ``[2I,-I,-I]`` (C1) / ``[I,-2I,2I,-I]`` (C2) rows. Returns
        ``(A, keys)`` where ``keys`` is an ordered list of ``(side, idx)`` selecting
        the control-point columns from ``{'s1': s1.points, 's2': s2.points}``.
        """
        d = self.degree
        fd = _fd_coeffs(k)

        def canon(side, idx):  # fold the shared junction point onto ('s1', d)
            return ("s1", d) if (side == "s2" and idx == 0) else (side, idx)

        coeffs: dict = {}
        order: list = []
        for j in range(k + 1):
            for side, idx, sign in (("s1", d - k + j, fd[j]), ("s2", j, -fd[j])):
                key = canon(side, idx)
                if key not in coeffs:
                    coeffs[key] = 0.0
                    order.append(key)
                coeffs[key] += sign
        keys = [key for key in order if abs(coeffs[key]) > 1e-15]
        A = np.hstack([coeffs[key] * eye for key in keys])
        return A, keys

    # ----------------------------------------- limits + continuity (symbolic)

    def _build_limits_and_continuity_symbolic(self) -> None:
        """Readable symbolic encoding (reference; also the path for non-HPoly sets).

        Builds the Bezier derivative curves explicitly (``s`` -> ``s'`` -> ...)
        and constrains each control point via Drake symbolic expressions /
        convex-set scaling constraints.
        """
        n_segs = len(self.untimed_segments)
        for s_idx, s in enumerate(self.untimed_segments):
            deriv = s
            for k in range(1, self.J + 1):
                deriv = deriv.derivative()
                start, stop = self._derivative_index_range(s_idx, k, n_segs)
                for i in range(start, stop):
                    self._add_set_membership(
                        self.deriv_sets[k - 1],
                        deriv.points[i],
                        self.T_powers[k],
                        f"deriv{k}_seg{s_idx}_cp{i}",
                    )

        for s_id in range(n_segs - 1):
            s1, s2 = self.untimed_segments[s_id], self.untimed_segments[s_id + 1]
            d1, d2 = s1, s2
            for k in range(1, self.continuity_order + 1):
                d1, d2 = d1.derivative(), d2.derivative()
                if s1.degree < k + 1:
                    continue
                c = self.prog.AddLinearConstraint(pd.eq(d1.points[-1], d2.points[0]))
                c.evaluator().set_description(f"C{k}_seg{s_id}_to_{s_id + 1}")

    def _add_set_membership(self, convex_set, point, scale, description) -> None:
        """Enforce ``point in scale * convex_set`` symbolically."""
        if isinstance(convex_set, pd.HPolyhedron):
            c = self.prog.AddLinearConstraint(
                pd.le(convex_set.A() @ point - convex_set.b() * scale, 0)
            )
            c.evaluator().set_description(description)
        else:
            aux = self.prog.NewContinuousVariables(self.dim)
            c = self.prog.AddLinearConstraint(pd.eq(aux, point))
            c.evaluator().set_description(description + "_aux")
            self.prog.SetInitialGuess(aux, np.zeros(self.dim))
            convex_set.AddPointInNonnegativeScalingConstraints(self.prog, aux, scale)

    def _derivative_index_range(self, s_idx: int, k: int, n_segs: int) -> Tuple[int, int]:
        """Control-point index range ``[start, stop)`` for the k-th derivative limit.

        The k-th derivative has ``d - k + 1`` control points. Its boundary control
        points (first of the first segment, last of the last segment) are pinned to
        zero *only when the k-th derivative itself rests*, i.e. ``terminal_order >=
        k`` — then they are redundant and skipped. When ``terminal_order < k`` the
        k-th derivative is free at the ends, so those boundary points MUST be
        constrained (skipping them would leave, e.g., snap unbounded at the
        boundary when only velocity/acceleration rest). Interior segments always
        skip the shared junction control point (index 0), which the previous
        segment owns. At ``k = 1, 2`` with the default ``terminal_order = 2`` this
        is exactly the historical velocity/acceleration range.
        """
        n_k = self.degree - k + 1
        is_first = s_idx == 0
        is_last = s_idx == n_segs - 1
        if self.terminal_order >= k:  # k-th derivative pinned to 0 at the ends
            start = 1 if is_first else 0
            stop = (n_k - 1) if is_last else n_k
        else:  # k-th derivative free at the ends -> constrain the boundary points
            start = 0 if is_first else 1
            stop = n_k
        return start, stop

    def _build_terminal_rest_constraints(self) -> None:
        # Zero derivatives 1 .. terminal_order at both ends. The j-th derivative
        # vanishes at the start iff the first j+1 control points coincide; pinning
        # pts[0] == pts[j] for j = 1 .. terminal_order achieves derivatives 1..m
        # (mirrored at the end). j = 1 is zero velocity, j = 2 zero acceleration, ...
        if self.terminal_order < 1:
            return
        first = self.untimed_segments[0]
        last = self.untimed_segments[-1]
        names = {1: "velocity", 2: "acceleration", 3: "jerk", 4: "snap"}
        for j in range(1, self.terminal_order + 1):
            tag = names.get(j, f"d{j}")
            c = self.prog.AddLinearConstraint(pd.eq(first.points[0], first.points[j]))
            c.evaluator().set_description(f"boundary_initial_{tag}_zero")
            c = self.prog.AddLinearConstraint(pd.eq(last.points[-1], last.points[-1 - j]))
            c.evaluator().set_description(f"boundary_terminal_{tag}_zero")

    # ------------------------------------------------------------------ solve

    def solve(
        self,
        new_planes: Dict[Tuple[int, int], Tuple[pb.BezierCurve, pb.BezierCurve]],
        tol: float = 1e-9,
    ) -> Tuple[pb.CompositeBezierCurve, float, float, float]:
        """Add ``new_planes`` and re-solve.

        ``new_planes`` maps ``(segment_id, obstacle_id)`` to a ``(a_curve,
        b_curve)`` pair describing the separating plane ``a(t).x + b(t) <= 0``.
        The constraint enforced on the trajectory is ``a(t).r(t) + b(t) <= tol``.

        Returns ``(trajectory, segment_time, optimal_cost, solver_time)``.
        """
        for (segment_id, obs_id), (a_curve, b_curve) in new_planes.items():
            self._add_plane_constraint(segment_id, obs_id, a_curve, b_curve, tol)

        result = self.solver.Solve(self.prog, solver_options=self._solver_opts)
        if not result.is_success():
            raise RuntimeError(f"Trajectory update failed: {result.get_solution_result()}")

        details: pd.ClarabelSolverDetails = result.get_solver_details()
        # Recover T from the minimized top power. The cost is T_powers[J], which the
        # solver drives tight; the lower powers are only lower-bounded by the rotated-cone
        # chain and can sit loose. That chain makes T_powers log-convex, so
        # T_powers[k]**(1/k) is non-decreasing in k -- taking the top, T_powers[J]**(1/J),
        # gives the largest, i.e. the one value of T that honors every derivative limit.
        Tp = result.GetSolution(self.T_powers)
        segment_time = float(Tp[-1] ** (1.0 / self.J))
        segments = [
            pb.BezierCurve(
                result.GetSolution(seg.points),
                initial_time=seg_id / self.N,
                final_time=(seg_id + 1) / self.N,
            )
            for seg_id, seg in enumerate(self.untimed_segments)
        ]
        trajectory = pb.CompositeBezierCurve(segments)
        return trajectory, segment_time, result.get_optimal_cost(), details.solve_time

    def _add_plane_constraint(self, segment_id, obs_id, a_curve, b_curve, tol) -> None:
        plane_degree = len(a_curve.points) - 1
        if self.use_symbolic_constraints:
            # Symbolic reference form: enforce the product curve's control points
            # (Bernstein convex-hull bound) directly.
            r_seg = pb.BezierCurve(self.untimed_segments[segment_id].points, 0, 1)
            a_seg = pb.BezierCurve(a_curve.points, 0, 1)
            b_seg = pb.BezierCurve(b_curve.points, 0, 1)
            prod = bezier_dot_product(a_seg, r_seg) + b_seg
            c = self.prog.AddLinearConstraint(pd.le(prod.points.flatten(), tol))
        else:
            A_mat, ub_vec = _plane_constraint_matrix(
                a_curve.points, b_curve.points.flatten(), self.degree, plane_degree, tol
            )
            lb = np.full(len(ub_vec), -np.inf)
            c = self.prog.AddLinearConstraint(A_mat, lb, ub_vec, self._seg_vars_flat[segment_id])
        c.evaluator().set_description(f"plane_obs{obs_id}_seg{segment_id}")
        self.added_plane_constraints.append(c)

    def clear_planes(self) -> None:
        """Remove all plane constraints added by previous :meth:`solve` calls."""
        for c in self.added_plane_constraints:
            self.prog.RemoveConstraint(c)
        self.added_plane_constraints = []
