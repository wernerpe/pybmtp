"""Derivative limit sets for the minimum-time planners.

A single :class:`Limits` object bundles the velocity / acceleration and the
optional jerk / snap convex limit sets, so every planner and updater takes one
``limits`` argument instead of a growing list of positional set arguments.

The number of *supplied* derivative sets (:attr:`Limits.order`, denoted ``J``)
determines exactly how many powers of the segment duration the trajectory-update
SOCP needs (velocity scales with ``T``, acceleration with ``T**2``, jerk with
``T**3``, snap with ``T**4``) -- the program is built no larger than that.

Only ``velocity`` is required: without a velocity bound the minimum time is
unbounded (T -> 0). ``acceleration``, ``jerk`` and ``snap`` are each optional and
must be contiguous from velocity up; omitting one leaves that derivative
degree-limited but keeps the problem well-posed. Velocity + acceleration is
byte-for-byte the previous acceleration-only program.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pydrake.all as pd


@dataclass(frozen=True)
class Limits:
    """Convex velocity/acceleration (and optional jerk/snap) limit sets.

    The sets must be *contiguous* from velocity upward: velocity and acceleration
    are required, ``jerk`` is optional, and ``snap`` may only be given together
    with ``jerk`` (there is no ``T**4`` limit without a ``T**3`` one). Every set
    must contain the origin.

    Parameters
    ----------
    velocity:
        Required convex set bounding the 1st derivative. Without it the minimum
        time is unbounded (T -> 0).
    acceleration, jerk, snap:
        Optional convex sets bounding the 2nd/3rd/4th derivatives, contiguous from
        velocity up (``jerk`` requires ``acceleration``; ``snap`` requires ``jerk``).
    """

    velocity: pd.ConvexSet
    acceleration: Optional[pd.ConvexSet] = None
    jerk: Optional[pd.ConvexSet] = None
    snap: Optional[pd.ConvexSet] = None

    def __post_init__(self) -> None:
        if self.velocity is None:
            raise ValueError(
                "Limits requires a velocity set: without a velocity bound the minimum time "
                "is unbounded (T -> 0)."
            )
        if self.jerk is not None and self.acceleration is None:
            raise ValueError(
                "Limits: a jerk set requires an acceleration set too -- derivative limits "
                "must be contiguous (no T**3 limit without a T**2 limit)."
            )
        if self.snap is not None and self.jerk is None:
            raise ValueError(
                "Limits: a snap set requires a jerk set too -- derivative limits must be "
                "contiguous (no T**4 limit without a T**3 limit)."
            )
        for name, s in (
            ("velocity", self.velocity),
            ("acceleration", self.acceleration),
            ("jerk", self.jerk),
            ("snap", self.snap),
        ):
            if s is None:
                continue
            if not s.PointInSet(np.zeros(s.ambient_dimension())):
                raise ValueError(f"the {name} set must contain the origin.")

    @property
    def sets(self) -> List[pd.ConvexSet]:
        """Supplied sets, ordered ``[velocity, (acceleration), (jerk), (snap)]``."""
        out = [self.velocity]
        for s in (self.acceleration, self.jerk, self.snap):
            if s is None:
                break
            out.append(s)
        return out

    @property
    def order(self) -> int:
        """Highest constrained derivative order ``J`` (1, 2, 3, or 4)."""
        return len(self.sets)

    def __len__(self) -> int:
        return self.order
