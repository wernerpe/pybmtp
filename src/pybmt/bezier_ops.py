"""Symbolic-friendly Bezier algebra used to build optimization constraints.

These helpers operate on Bezier control points that may be Drake decision
variables (``pydrake.symbolic.Variable``), not just floats. That is the whole
reason pybmt depends on ``pybezier``: it lets us express a constraint like
``a(t) . r(t) + b(t) <= tol`` symbolically over the trajectory's control points
and then hand the resulting affine expressions to a MathematicalProgram.

The same quantities can be assembled in a faster numeric matrix form (see the
updater modules); these symbolic routines are the readable reference that the
matrix forms are derived from and validated against.
"""

from __future__ import annotations

import numpy as np
import pybezier as pb
from scipy.special import comb


def binomial(n: int, k: int) -> int:
    """Exact binomial coefficient C(n, k)."""
    return comb(n, k, exact=True)


def bezier_dot_product(a_curve: pb.BezierCurve, r_curve: pb.BezierCurve) -> pb.BezierCurve:
    """Scalar Bezier curve ``a(t) . r(t)`` for two vector-valued curves.

    Uses the Farouki-Rajan product formula. ``a_curve`` may carry symbolic
    (Drake variable) control points; ``r_curve`` is typically numeric. The
    result is a degree ``(m + n)`` scalar curve whose control points are the
    affine expressions
        c_i = sum_{j} [C(m,j) C(n,i-j) / C(m+n,i)] * (a_j . r_{i-j}).
    By the Bernstein convex-hull property, ``c_i <= tol`` for all i implies
    ``a(t) . r(t) <= tol`` for all t.
    """
    m = a_curve.degree
    n = r_curve.degree
    degree = m + n

    points = np.zeros((degree + 1, 1), dtype=object)
    for i in range(degree + 1):
        for j in range(max(0, i - n), min(m, i) + 1):
            k = i - j
            coeff = binomial(m, j) * binomial(n, k)
            points[i, 0] += np.dot(a_curve.points[j], r_curve.points[k]) * coeff
        points[i, 0] /= binomial(degree, i)

    return pb.BezierCurve(points, a_curve.initial_time, a_curve.final_time)
