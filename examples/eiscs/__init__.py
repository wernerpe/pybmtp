"""EISCS baseline: Edge-Inflation regions + scsplanning (the paper's SCS baseline).

This is deliberately NOT part of the installable ``pybmtp`` package: it depends
on ``scsplanning``, which is not published to PyPI. It lives with the experiments
and is imported directly by the example scripts and tests::

    from eiscs import EISCSPlanner, solve_scs_minimum_time, SCSResult
"""

from .result import SCSResult
from .scs import EISCSPlanner, solve_scs_minimum_time

__all__ = ["EISCSPlanner", "SCSResult", "solve_scs_minimum_time"]
