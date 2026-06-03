"""Trajectory-update SOCP for the biconvex minimum-time planner.

Given a fixed set of separating planes (one per active trajectory-segment /
obstacle pair), this solves for the Bezier control points and the common
segment duration ``T`` that minimize total time while satisfying:

  * boundary conditions (position, and optionally zero velocity/acceleration),
  * velocity and acceleration limits (convex-set membership of the Bezier
    derivative control points, scaled by powers of ``T``),
  * C1 / C2 continuity between segments,
  * the half-space constraints from the current planes.

All segments share one duration ``T`` (the time grid is uniform); the rotated
Lorentz "power ladder" links ``T_powers = [1, T, T^2]`` so velocity (linear in
``T``) and acceleration (linear in ``T^2``) limits stay convex.

Two equivalent constraint encodings live here:

  * a **symbolic** form built from ``pydrake`` expressions and Bezier
    derivative curves — readable, and the reference implementation;
  * a **fast matrix** form that writes the same affine constraints directly as
    ``AddLinearConstraint(A, lb, ub, vars)``, skipping Drake's symbolic-formula
    parsing (which dominated build time, ~7 ms per construction).

Every matrix block below is annotated with the symbolic form it expands. The
``use_symbolic_constraints`` flag forces the symbolic path even for
HPolyhedron sets, which the test-suite uses to assert the two forms produce
identical solutions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pybezier as pb
import pydrake.all as pd
from scipy.special import comb as _comb

from ..bezier_ops import bezier_dot_product


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

    Construction wires up the time-invariant constraints (boundary, velocity,
    acceleration, continuity); planes are added per :meth:`solve` call and can
    be removed with :meth:`clear_planes`, so the same program object is reused
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
    velocity_set, acceleration_set:
        Convex limit sets. Both must contain the origin.
    degree:
        Bezier degree of each trajectory segment.
    add_terminal_velocity_constraint, add_terminal_acceleration_constraint:
        Pin start/end velocity / acceleration to zero.
    add_c2_continuity:
        Enforce C2 (acceleration) continuity at segment junctions in addition
        to C1.
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
        velocity_set: pd.ConvexSet,
        acceleration_set: pd.ConvexSet,
        degree: int = 6,
        add_terminal_velocity_constraint: bool = True,
        add_terminal_acceleration_constraint: bool = True,
        add_c2_continuity: bool = True,
        use_symbolic_constraints: bool = False,
    ):
        self.source = np.asarray(source, float)
        self.target = np.asarray(target, float)
        self.dim = len(self.source)
        self.N = num_segments
        self.degree = degree
        self.vel_set = velocity_set
        self.acc_set = acceleration_set
        self.add_terminal_velocity_constraint = add_terminal_velocity_constraint
        self.add_terminal_acceleration_constraint = add_terminal_acceleration_constraint
        self.add_c2_continuity = add_c2_continuity
        self.domain = domain

        hpoly_sets = isinstance(velocity_set, pd.HPolyhedron) and isinstance(
            acceleration_set, pd.HPolyhedron
        )
        self.use_symbolic_constraints = use_symbolic_constraints or not hpoly_sets

        assert self.vel_set.PointInSet(np.zeros(self.dim)), "0 must be in the velocity set"
        assert self.acc_set.PointInSet(np.zeros(self.dim)), "0 must be in the acceleration set"

        self.solver = pd.ClarabelSolver()
        self.prog = pd.MathematicalProgram()
        self._solver_opts = pd.SolverOptions()
        self._solver_opts.SetOption(pd.CommonSolverOption.kPrintToConsole, False)
        self.added_plane_constraints: list = []

        self._build_time_variables()
        self._build_segments()
        self._build_boundary_constraints()
        if self.use_symbolic_constraints:
            self._build_limit_and_continuity_symbolic()
        else:
            self._build_limit_and_continuity_matrix()
        self._build_terminal_rest_constraints()

    # ------------------------------------------------------------------ setup

    def _build_time_variables(self) -> None:
        # T_powers = [1, T, T^2]. The rotated Lorentz cone T_powers[0]*T_powers[2]
        # >= T_powers[1]^2 with T_powers[0] == 1 enforces T_powers[2] >= T^2,
        # keeping the (otherwise non-convex) power relationship convex. Velocity
        # limits scale with T (T_powers[1]); acceleration with T^2 (T_powers[2]).
        self.J = 2
        self.T_powers = self.prog.NewContinuousVariables(self.J + 1, "T_powers")
        self.prog.AddLinearEqualityConstraint(self.T_powers[0] == 1)
        for j in range(2, self.J + 1):
            A = np.zeros((3, self.J + 1))
            A[0, j - 2] = 1
            A[1, j] = 1
            A[2, j - 1] = 1
            self.prog.AddRotatedLorentzConeConstraint(A=A, b=np.zeros(3), vars=self.T_powers)
        self.prog.AddLinearCost(self.T_powers[-1])

    def _build_segments(self) -> None:
        # Control points are decision variables, except shared junction points
        # (C0 continuity: a segment's first point IS the previous segment's
        # last point) and the boundary-locked endpoint triples.
        self.untimed_segments: List[pb.BezierCurve] = []
        n_cp = self.degree + 1
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

                # First/last 3 control points are pinned to source/target by the
                # zero boundary vel+acc constraints, so they skip the domain box.
                if seg_id == 0 and i < 3:
                    continue
                if seg_id == self.N - 1 and i >= n_cp - 3:
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

    def _build_limit_and_continuity_matrix(self) -> None:
        """Fast numeric matrix encoding of velocity/acceleration/continuity.

        Bezier derivative identities (control-point finite differences):
            sdot[i]  = d   * (pts[i+1] - pts[i])
            sddot[i] = d(d-1) * (pts[i+2] - 2 pts[i+1] + pts[i])
        """
        d = self.degree
        n_segs = len(self.untimed_segments)
        vel_A, vel_b = self.vel_set.A(), self.vel_set.b()
        acc_A, acc_b = self.acc_set.A(), self.acc_set.b()
        eye = np.eye(self.dim)

        # Velocity:  vel_A @ sdot[i] <= vel_b * T
        #   symbolic: pd.le(vel_A @ (d*(pts[i+1]-pts[i])) - vel_b*T_powers[1], 0)
        #   expand  : [ d*vel_A | -d*vel_A | -vel_b ] @ [pts[i+1]; pts[i]; T] <= 0
        A_vel = np.hstack([d * vel_A, -d * vel_A, -vel_b.reshape(-1, 1)])
        lb_vel = np.full(vel_A.shape[0], -np.inf)
        ub_vel = np.zeros(vel_A.shape[0])

        # Acceleration:  acc_A @ sddot[i] <= acc_b * T^2,  c := d(d-1)
        #   symbolic: pd.le(acc_A @ (c*(pts[i+2]-2pts[i+1]+pts[i])) - acc_b*T_powers[2], 0)
        #   expand  : [ c*acc_A | -2c*acc_A | c*acc_A | -acc_b ] @ [pts[i+2]; pts[i+1]; pts[i]; T^2] <= 0
        c_acc = d * (d - 1)
        A_acc = np.hstack([c_acc * acc_A, -2 * c_acc * acc_A, c_acc * acc_A, -acc_b.reshape(-1, 1)])
        lb_acc = np.full(acc_A.shape[0], -np.inf)
        ub_acc = np.zeros(acc_A.shape[0])

        # C1 continuity:  s1dot[-1] == s2dot[0]. With C0 (s2.pts[0] == s1.pts[-1]):
        #   symbolic: pd.eq(d*(s1.pts[-1]-s1.pts[-2]), d*(s2.pts[1]-s1.pts[-1]))
        #   expand  : [ 2I | -I | -I ] @ [s1.pts[-1]; s1.pts[-2]; s2.pts[1]] == 0
        A_c1 = np.hstack([2 * eye, -eye, -eye])
        eq_c1 = np.zeros(self.dim)

        # C2 continuity:  s1ddot[-1] == s2ddot[0]. With C0, dividing by c:
        #   symbolic: pd.eq(s1.pts[-1]-2s1.pts[-2]+s1.pts[-3], s2.pts[2]-2s2.pts[1]+s1.pts[-1])
        #   expand  : [ I | -2I | 2I | -I ] @ [s1.pts[-3]; s1.pts[-2]; s2.pts[1]; s2.pts[2]] == 0
        A_c2 = np.hstack([eye, -2 * eye, 2 * eye, -eye])
        eq_c2 = np.zeros(self.dim)

        for s_idx, s in enumerate(self.untimed_segments):
            pts = s.points
            v_start, v_stop, a_start, a_stop = self._derivative_index_range(s_idx, n_segs)
            for i in range(v_start, v_stop):
                c = self.prog.AddLinearConstraint(
                    A_vel, lb_vel, ub_vel, np.concatenate([pts[i + 1], pts[i], [self.T_powers[1]]])
                )
                c.evaluator().set_description(f"velocity_seg{s_idx}_cp{i}")
            for i in range(a_start, a_stop):
                c = self.prog.AddLinearConstraint(
                    A_acc,
                    lb_acc,
                    ub_acc,
                    np.concatenate([pts[i + 2], pts[i + 1], pts[i], [self.T_powers[2]]]),
                )
                c.evaluator().set_description(f"acceleration_seg{s_idx}_cp{i}")

        for s_id in range(n_segs - 1):
            s1, s2 = self.untimed_segments[s_id], self.untimed_segments[s_id + 1]
            if s1.degree >= 2:
                c = self.prog.AddLinearConstraint(
                    A_c1, eq_c1, eq_c1, np.concatenate([s1.points[-1], s1.points[-2], s2.points[1]])
                )
                c.evaluator().set_description(f"C1_seg{s_id}_to_{s_id+1}")
            if self.add_c2_continuity and s1.degree >= 3:
                c = self.prog.AddLinearConstraint(
                    A_c2,
                    eq_c2,
                    eq_c2,
                    np.concatenate([s1.points[-3], s1.points[-2], s2.points[1], s2.points[2]]),
                )
                c.evaluator().set_description(f"C2_seg{s_id}_to_{s_id+1}")

    # ----------------------------------------- limits + continuity (symbolic)

    def _build_limit_and_continuity_symbolic(self) -> None:
        """Readable symbolic encoding (reference; also the path for non-HPoly sets).

        This is the form the matrix version above expands. It builds the Bezier
        derivative curves explicitly and constrains each control point via
        Drake symbolic expressions / convex-set scaling constraints.
        """
        n_segs = len(self.untimed_segments)
        for s_idx, s in enumerate(self.untimed_segments):
            sdot = s.derivative()
            sddot = sdot.derivative()
            v_start, v_stop, a_start, a_stop = self._derivative_index_range(s_idx, n_segs)

            for i in range(v_start, v_stop):
                self._add_set_membership(
                    self.vel_set, sdot.points[i], self.T_powers[1], f"velocity_seg{s_idx}_cp{i}"
                )
            for i in range(a_start, a_stop):
                self._add_set_membership(
                    self.acc_set,
                    sddot.points[i],
                    self.T_powers[2],
                    f"acceleration_seg{s_idx}_cp{i}",
                )

        for s_id in range(n_segs - 1):
            s1, s2 = self.untimed_segments[s_id], self.untimed_segments[s_id + 1]
            s1dot, s2dot = s1.derivative(), s2.derivative()
            if s1.degree >= 2:
                c = self.prog.AddLinearConstraint(pd.eq(s1dot.points[-1], s2dot.points[0]))
                c.evaluator().set_description(f"C1_seg{s_id}_to_{s_id+1}")
            if self.add_c2_continuity and s1.degree >= 3:
                s1ddot, s2ddot = s1dot.derivative(), s2dot.derivative()
                c = self.prog.AddLinearConstraint(pd.eq(s1ddot.points[-1], s2ddot.points[0]))
                c.evaluator().set_description(f"C2_seg{s_id}_to_{s_id+1}")

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

    def _derivative_index_range(self, s_idx: int, n_segs: int) -> Tuple[int, int, int, int]:
        """Control-point index ranges for velocity/acceleration constraints.

        The boundary control points of the first/last segment are pinned by the
        zero-velocity/zero-acceleration boundary constraints, so we skip them
        here to avoid redundant (and over-tight) rows. Interior segments share
        their first control point with the previous segment (C0), so that index
        is also skipped.
        """
        d = self.degree
        is_first = s_idx == 0
        is_last = s_idx == n_segs - 1
        if self.add_terminal_velocity_constraint or self.add_terminal_acceleration_constraint:
            v_start = 1 if is_first else 0
            v_stop = d - 1 if is_last else d
            a_start = 1 if is_first else 0
            a_stop = d - 2 if is_last else d - 1
        else:
            v_start = 0 if is_first else 1
            v_stop = d
            a_start = 0 if is_first else 1
            a_stop = d - 1
        return v_start, v_stop, a_start, a_stop

    def _build_terminal_rest_constraints(self) -> None:
        if self.add_terminal_velocity_constraint:
            # Zero endpoint velocity <=> first two (last two) control points coincide.
            c = self.prog.AddLinearConstraint(
                pd.eq(self.untimed_segments[0].points[0], self.untimed_segments[0].points[1])
            )
            c.evaluator().set_description("boundary_initial_velocity_zero")
            c = self.prog.AddLinearConstraint(
                pd.eq(self.untimed_segments[-1].points[-1], self.untimed_segments[-1].points[-2])
            )
            c.evaluator().set_description("boundary_terminal_velocity_zero")
        if self.untimed_segments[0].degree > 3 and self.add_terminal_acceleration_constraint:
            # Zero endpoint acceleration <=> first/last three control points colinear-at-rest.
            c = self.prog.AddLinearConstraint(
                pd.eq(self.untimed_segments[0].points[0], self.untimed_segments[0].points[2])
            )
            c.evaluator().set_description("boundary_initial_acceleration_zero")
            c = self.prog.AddLinearConstraint(
                pd.eq(self.untimed_segments[-1].points[-1], self.untimed_segments[-1].points[-3])
            )
            c.evaluator().set_description("boundary_terminal_acceleration_zero")

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
        segment_time = float(np.sqrt(result.GetSolution(self.T_powers)[2]))
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
