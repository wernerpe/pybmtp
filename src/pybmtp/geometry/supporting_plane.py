"""Supporting-plane constraints for convex obstacles, via conic duality.

A separating plane ``{x : a.x + b = 0}`` supports an obstacle ``O`` (i.e. lies
on one side of it) iff ``a.x + b >= step_back`` for every ``x`` in ``O``. Rather
than enumerate the obstacle's points, we use conic duality: writing ``O`` as the
feasible set of a conic program, the supporting condition becomes a finite set
of constraints on a dual variable ``nu``. This makes the supporting-plane test a
small linear/second-order-cone feasibility problem regardless of how the
obstacle is described.

Supported obstacle types: ``HPolyhedron``, ``Hyperellipsoid``, ``VPolytope``,
and ``Hyperrectangle`` (normalized to ``HPolyhedron``).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pydrake.all as pd

Obstacle = Union[pd.HPolyhedron, pd.Hyperellipsoid, pd.Hyperrectangle, pd.VPolytope]


def get_conic_description(
    obstacle: Obstacle,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, str]:
    """Conic description ``(A, B, c, cone_type)`` of a convex obstacle.

    The obstacle is written as
        O = { x : there exists a "membership certificate" feasible in the cone },
    and the returned tuple feeds :func:`add_supporting_plane`, which uses the
    dual cone to enforce that a plane supports ``O``.

    Returns
    -------
    A : (k, dim) array       coupling the plane normal to the dual variable.
    B : (k, p) array or None  extra equality coupling (used by VPolytope).
    c : (k,) or (k, 1) array  offset feeding the plane's scalar offset bound.
    cone_type : {"nonneg_orthant", "soc"}.
    """
    if isinstance(obstacle, pd.Hyperrectangle):
        obstacle = obstacle.MakeHPolyhedron()

    if isinstance(obstacle, pd.HPolyhedron):
        # {x : A_h x <= b_h}; dual lives in the nonnegative orthant.
        return -obstacle.A(), None, obstacle.b(), "nonneg_orthant"

    if isinstance(obstacle, pd.Hyperellipsoid):
        # {x : ||A_e (x - center)|| <= 1}; supported via a second-order cone.
        Ae = obstacle.A()
        A_conic = np.zeros((Ae.shape[0] + 1, Ae.shape[1]))
        b_conic = np.zeros((Ae.shape[0] + 1, 1))
        A_conic[1:, :] = Ae
        b_conic[0] = 1
        b_conic[1:, 0] = -(Ae @ obstacle.center())
        return A_conic, None, b_conic, "soc"

    if isinstance(obstacle, pd.VPolytope):
        # conv(vertices); membership as a convex combination, dual in the orthant.
        P = obstacle.vertices()
        dim = obstacle.ambient_dimension()
        A_conic = np.zeros((2 * dim + P.shape[1] + 2, dim))
        A_conic[:dim, :dim] = np.eye(dim)
        A_conic[dim : 2 * dim, :dim] = -np.eye(dim)
        B_conic = np.concatenate(
            (-P, P, np.eye(P.shape[1]), np.ones((1, P.shape[1])), -np.ones((1, P.shape[1]))), axis=0
        )
        b_conic = np.zeros((2 * dim + P.shape[1] + 2,))
        b_conic[-2] = -1
        b_conic[-1] = 1
        return A_conic, B_conic, b_conic, "nonneg_orthant"

    raise NotImplementedError(f"Unsupported obstacle type: {type(obstacle)}")


def add_supporting_plane(
    prog: pd.MathematicalProgram,
    obstacle: Obstacle,
    name: str,
    norm_val: float = 1.0,
    restrict_norm: bool = True,
    step_back: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add variables ``(a, b)`` for a plane that supports ``obstacle``.

    Enforces ``a.x + b >= step_back`` for all ``x`` in the obstacle, encoded via
    the conic dual (so no point enumeration). If ``restrict_norm`` is True, also
    constrains ``||a|| <= norm_val`` (a second-order cone) to keep the plane
    well-scaled.

    Returns the decision-variable arrays ``a`` (shape ``(dim,)``) and ``b``
    (shape ``(1,)``).
    """
    dim = obstacle.ambient_dimension()

    A_conic, B_conic, b_conic, cone_type = get_conic_description(obstacle)
    nu = prog.NewContinuousVariables(A_conic.shape[0], "nu_" + name)
    a = prog.NewContinuousVariables(dim, "a_" + name)
    b = prog.NewContinuousVariables(1, "b_" + name)

    # Dual coupling: a = A^T nu, optional B^T nu = 0, and b >= c^T nu + step_back.
    prog.AddLinearConstraint(pd.eq(a, A_conic.T @ nu))
    if B_conic is not None:
        prog.AddLinearConstraint(pd.eq(B_conic.T @ nu, np.zeros(B_conic.shape[1])))
    prog.AddLinearConstraint(pd.ge(b, b_conic.T @ nu + step_back))

    if cone_type == "soc":
        prog.AddLorentzConeConstraint(nu)
    elif cone_type == "nonneg_orthant":
        prog.AddBoundingBoxConstraint(0, np.inf, nu)
    else:
        raise NotImplementedError(f"cone type not implemented: {cone_type}")

    if restrict_norm:
        # ||a|| <= norm_val via the Lorentz cone on z = [norm_val, a].
        z = prog.NewContinuousVariables(dim + 1, "z_" + name)
        prog.AddLinearConstraint(norm_val == z[0])
        prog.AddLinearConstraint(pd.eq(a, z[1:]))
        prog.AddLorentzConeConstraint(z)

    return a, b
