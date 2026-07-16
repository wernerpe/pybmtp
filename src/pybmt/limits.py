"""Derivative limit sets for the minimum-time planners.

A single :class:`Limits` object bundles the velocity / acceleration and the
optional jerk / snap convex limit sets, so every planner and updater takes one
``limits`` argument instead of a growing list of positional set arguments.

The number of *supplied* derivative sets (:attr:`Limits.order`, denoted ``J``)
determines exactly how many powers of the segment duration the trajectory-update
SOCP needs (velocity scales with ``T``, acceleration with ``T**2``, jerk with
``T**3``, snap with ``T**4``) -- the program is built no larger than that. Give
only velocity + acceleration and it is byte-for-byte the acceleration-only
program; add ``jerk`` and/or ``snap`` and the ``T_powers`` ladder grows to match.
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
    velocity, acceleration:
        Required convex sets bounding the 1st and 2nd derivatives.
    jerk, snap:
        Optional convex sets bounding the 3rd and 4th derivatives.
    """

    velocity: pd.ConvexSet
    acceleration: pd.ConvexSet
    jerk: Optional[pd.ConvexSet] = None
    snap: Optional[pd.ConvexSet] = None

    def __post_init__(self) -> None:
        if self.velocity is None or self.acceleration is None:
            raise ValueError("Limits requires both a velocity and an acceleration set.")
        if self.snap is not None and self.jerk is None:
            raise ValueError(
                "Limits: a snap set requires a jerk set too -- derivative limits must be "
                "contiguous (there is no T**4 limit without a T**3 limit)."
            )
        for name, s in zip(("velocity", "acceleration", "jerk", "snap"), self.sets + [None]):
            if s is None:
                continue
            if not s.PointInSet(np.zeros(s.ambient_dimension())):
                raise ValueError(f"the {name} set must contain the origin.")

    @property
    def sets(self) -> List[pd.ConvexSet]:
        """Supplied sets, ordered ``[velocity, acceleration, (jerk), (snap)]``."""
        out = [self.velocity, self.acceleration]
        if self.jerk is not None:
            out.append(self.jerk)
        if self.snap is not None:
            out.append(self.snap)
        return out

    @property
    def order(self) -> int:
        """Highest constrained derivative order ``J`` (2, 3, or 4)."""
        return len(self.sets)

    def __len__(self) -> int:
        return self.order
