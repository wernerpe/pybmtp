"""Polygonal warm start for the biconvex minimum-time planners.

Connects consecutive waypoints with straight, rest-to-rest, minimum-time Bezier
segments stitched onto one timeline. Each segment is a single-segment, plane-free
:class:`~pybmt.updaters.TrajectoryUpdater` solve under the *same* limits,
continuity and terminal-rest the final trajectory optimization uses, so the warm
start already satisfies the constraints the planner will enforce.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pybezier as pb
import pydrake.all as pd

from .limits import Limits
from .updaters import TrajectoryUpdater

__all__ = ["polygonal_initialization"]


def polygonal_initialization(
    waypoints: np.ndarray,
    domain: pd.HPolyhedron,
    limits: Limits,
    degree: int = 6,
    continuity_order: Optional[int] = None,
    terminal_order: Optional[int] = None,
    on_segment: bool = True,
    force_same_segment_time: bool = False,
) -> pb.CompositeBezierCurve:
    """Straight, rest-to-rest, minimum-time warm start through ``waypoints``.

    ``waypoints`` is an ``(N, dim)`` array whose first and last entries are the
    start and goal. Each consecutive pair is joined by one minimum-time Bezier
    segment of the given ``degree`` that respects ``limits`` (velocity, and
    optionally acceleration / jerk / snap) and rests at both ends. With
    ``force_same_segment_time`` every segment is stretched to the slowest one's
    duration, yielding the uniform time grid the biconvex updater warm-starts from.

    ``on_segment`` (default) confines each segment to the straight line between its
    waypoints, so a collision-free input polyline yields a collision-free seed.

    ``degree``, ``continuity_order`` and ``terminal_order`` should match the final
    trajectory optimization so the seed satisfies the same constraints.
    """
    waypoints = np.asarray(waypoints, float)
    if waypoints.ndim != 2 or len(waypoints) < 2:
        raise ValueError("waypoints must be an (N, dim) array with N >= 2")

    control_points, durations = [], []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        if np.linalg.norm(end - start) < 1e-9:  # coincident waypoints -> a zero-length hold
            control_points.append(np.tile(start, (degree + 1, 1)))
            durations.append(1e-6)
            continue
        updater = TrajectoryUpdater(
            start,
            end,
            domain,
            1,
            limits,
            degree=degree,
            continuity_order=continuity_order,
            terminal_order=terminal_order,
            on_segment=on_segment,  # keep the warm start on the (collision-free) waypoint segment
        )
        curve, segment_time, _, _ = updater.solve({})
        control_points.append(curve.curves[0].points)
        durations.append(segment_time)

    if force_same_segment_time:
        durations = [max(durations)] * len(durations)

    segments, t = [], 0.0
    for pts, dur in zip(control_points, durations):
        segments.append(pb.BezierCurve(pts, t, t + dur))
        t += dur
    return pb.CompositeBezierCurve(segments)
