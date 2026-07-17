"""Result object for the EISCS baseline planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pybezier as pb
import pydrake.all as pd


@dataclass
class SCSResult:
    """Outcome of an :class:`~eiscs.scs.EISCSPlanner` solve.

    Attributes
    ----------
    trajectory:
        The final time-scaled trajectory (physical seconds).
    total_time:
        Total trajectory duration.
    regions:
        The collision-free convex regions the trajectory was planned through
        (after near-zero-length pruning and Assumption-1 splitting).
    num_regions:
        Number of regions actually solved through (post-pruning).
    solver_type:
        Which scsplanning backend produced the trajectory (``"biconvex"``,
        ``"polygonal"``, or ``"nonconvex"``).
    timing:
        Wall-clock breakdown (keys: ``total``, ``region_generation``,
        ``trajectory_optimization``).
    """

    trajectory: pb.CompositeBezierCurve
    total_time: float
    regions: List[pd.HPolyhedron] = field(default_factory=list)
    num_regions: int = 0
    solver_type: str = "biconvex"
    timing: Dict[str, float] = field(default_factory=dict)
