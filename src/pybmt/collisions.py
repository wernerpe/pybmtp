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
    "intersect_with_hyperspheres",
    "intersect_with_obstacles",
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


def _sphere_center_radius(sphere) -> tuple[np.ndarray, float]:
    """Extract ``(center, radius)`` from a Drake hypersphere or a raw tuple.

    The native kernel only handles *balls* (a center and a scalar radius). A
    Drake ``Hyperellipsoid`` is the ball ``{x : ||A(x - c)|| <= 1}`` only when
    ``A = (1/r) I`` (isotropic); anisotropic ellipsoids are rejected rather than
    silently mis-checked.
    """
    if isinstance(sphere, pd.Hyperellipsoid):
        A = np.asarray(sphere.A(), dtype=float)
        c = np.asarray(sphere.center(), dtype=float).reshape(-1)
        diag = np.diag(A)
        if not (np.allclose(A, np.diag(diag)) and np.allclose(diag, diag[0])
                and diag[0] > 0):
            raise ValueError(
                "intersect_with_hyperspheres only supports isotropic "
                "Hyperellipsoids (A = (1/r) I, i.e. true hyperspheres); got an "
                "anisotropic ellipsoid.")
        return c, float(1.0 / diag[0])
    center, radius = sphere
    return np.asarray(center, dtype=float).reshape(-1), float(radius)


def intersect_with_hyperspheres(
    curve: pb.CompositeBezierCurve,
    spheres: Sequence,
    ignore: Mapping[int, Iterable[int]] | None = None,
    tol: float = 1e-2,
    parallelize: bool = True,
) -> dict[int, list[int]]:
    """Per-segment intersection test against ball obstacles.

    ``spheres`` is a sequence of either Drake ``Hyperellipsoid`` hyperspheres or
    ``(center, radius)`` pairs. The return value matches
    :func:`intersect_with_hpolyhedra`: ``{segment_index: [sphere_indices]}`` for
    every segment not proven collision-free (free segments are omitted, so an
    empty dict means the whole trajectory is clear).
    """
    centers, radii = [], []
    for s in spheres:
        c, r = _sphere_center_radius(s)
        centers.append(c)
        radii.append(r)

    # The native sphere kernel takes one ignore-list per segment (not a dict)
    # and returns a per-segment list (not a dict), so translate both ways.
    num_segments = len(curve.curves)
    ignore_lists: list[list[int]] = [[] for _ in range(num_segments)]
    for seg, obs in (ignore or {}).items():
        ignore_lists[int(seg)] = [int(o) for o in obs]

    per_segment = _native.intersect_composite_bezier_with_spheres(
        to_native_composite(curve), centers, radii, ignore_lists, tol,
        parallelize)
    return {seg: [int(o) for o in obs]
            for seg, obs in enumerate(per_segment) if obs}


def intersect_with_obstacles(
    curve: pb.CompositeBezierCurve,
    obstacles: Sequence[pd.ConvexSet],
    ignore: Mapping[int, Iterable[int]] | None = None,
    tol: float = 1e-2,
    parallelize: bool = True,
) -> dict[int, list[int]]:
    """Per-segment intersection test against a mixed list of convex obstacles.

    ``obstacles`` is a single ordered list of Drake ``ConvexSet`` obstacles;
    each is dispatched to the matching kernel (``HPolyhedron`` -> polytope path,
    isotropic ``Hyperellipsoid`` -> sphere path). The list order *is* the
    obstacle index space: both ``ignore`` and the returned
    ``{segment_index: [obstacle_indices]}`` use those global indices, so callers
    never see the per-kernel batching. Anything that is neither an
    ``HPolyhedron`` nor a (true) hypersphere raises.
    """
    ignore = ignore or {}
    hpolys, hpoly_global = [], []
    spheres, sphere_global = [], []
    for gi, obs in enumerate(obstacles):
        if isinstance(obs, pd.HPolyhedron):
            hpolys.append(obs)
            hpoly_global.append(gi)
        elif isinstance(obs, pd.Hyperellipsoid):
            spheres.append(obs)
            sphere_global.append(gi)
        else:
            raise TypeError(
                f"obstacle {gi} has unsupported type {type(obs).__name__}; "
                "expected HPolyhedron or an isotropic Hyperellipsoid")

    def _local_ignore(global_to_local: dict[int, int]) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for seg, obs_ids in ignore.items():
            locs = [global_to_local[o] for o in obs_ids if o in global_to_local]
            if locs:
                out[int(seg)] = locs
        return out

    result: dict[int, list[int]] = {}
    if hpolys:
        local = intersect_with_hpolyhedra(
            curve, hpolys, ignore=_local_ignore({g: i for i, g in enumerate(hpoly_global)}),
            tol=tol, parallelize=parallelize)
        for seg, obs in local.items():
            result.setdefault(seg, []).extend(hpoly_global[o] for o in obs)
    if spheres:
        local = intersect_with_hyperspheres(
            curve, spheres, ignore=_local_ignore({g: i for i, g in enumerate(sphere_global)}),
            tol=tol, parallelize=parallelize)
        for seg, obs in local.items():
            result.setdefault(seg, []).extend(sphere_global[o] for o in obs)

    return {seg: sorted(obs) for seg, obs in sorted(result.items())}
