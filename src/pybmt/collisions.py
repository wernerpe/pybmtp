"""Bezier-curve collision checking against convex obstacles.

This is the Python face of the vendored C++ kernel (``pybmt._native``). The
kernel works on its own lightweight ``BezierCurve`` / ``CompositeBezierCurve``
representation, so the helpers here round-trip ``pybezier`` curves into native
curves (a cheap control-point + time-interval copy) and hand off to the
parallel intersection routine.

The kernel is *conservative*: it subdivides each segment with De Casteljau and
uses the Bernstein convex-hull property, so it never reports a curve that
actually touches an obstacle as collision-free. The price is occasional false
collisions on curves that pass within ``tol`` of an obstacle face.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pybezier as pb
import pydrake.all as pd

from . import _native

__all__ = [
    "to_native_curve",
    "to_native_composite",
    "curve_is_collision_free",
    "intersect_with_hpolyhedra",
]


def to_native_curve(curve: pb.BezierCurve) -> "_native.BezierCurve":
    """Copy a ``pybezier.BezierCurve`` into a native ``BezierCurve``."""
    return _native.BezierCurve(
        np.asarray(curve.points, dtype=float),
        float(curve.initial_time),
        float(curve.final_time),
    )


def to_native_composite(
    curve: pb.CompositeBezierCurve,
) -> "_native.CompositeBezierCurve":
    """Copy a ``pybezier.CompositeBezierCurve`` into a native composite curve."""
    return _native.CompositeBezierCurve([to_native_curve(c) for c in curve.curves])


def _hpolyhedron_arrays(
    obstacles: Sequence[pd.HPolyhedron],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split a list of HPolyhedra into their ``A`` and ``b`` arrays."""
    As = [np.asarray(o.A(), dtype=float) for o in obstacles]
    bs = [np.asarray(o.b(), dtype=float).reshape(-1) for o in obstacles]
    return As, bs


def curve_is_collision_free(
    curve: pb.BezierCurve,
    obstacle: pd.HPolyhedron,
    tol: float = 1e-2,
) -> bool:
    """Return ``True`` iff ``curve`` is proven to avoid ``obstacle``.

    Conservative: a ``True`` result is a guarantee, a ``False`` result may be a
    near-miss within ``tol``.
    """
    A = np.asarray(obstacle.A(), dtype=float)
    b = np.asarray(obstacle.b(), dtype=float).reshape(-1)
    return bool(_native.bezier_curve_hpolyhedron_collision_free(
        to_native_curve(curve), A, b, tol))


def intersect_with_hpolyhedra(
    curve: pb.CompositeBezierCurve,
    obstacles: Sequence[pd.HPolyhedron],
    ignore: Mapping[int, Iterable[int]] | None = None,
    tol: float = 1e-2,
    parallelize: bool = True,
) -> dict[int, list[int]]:
    """Per-segment obstacle intersection test for a composite curve.

    Returns ``{segment_index: [obstacle_indices]}`` for every segment that is
    not proven collision-free against the listed obstacles. Segments proven
    free are omitted, so an empty dict means the whole trajectory is clear.

    ``ignore`` maps a segment index to obstacle indices to skip for that segment
    (e.g. the obstacle a segment is allowed to touch at its grasp endpoint).
    """
    As, bs = _hpolyhedron_arrays(obstacles)
    ignore_map = {int(k): [int(i) for i in v] for k, v in (ignore or {}).items()}
    hits = _native.intersect_composite_bezier_with_hpolyhedra(
        to_native_composite(curve), As, bs, ignore_map, tol, parallelize)
    return {int(seg): [int(o) for o in obs] for seg, obs in hits.items()}
