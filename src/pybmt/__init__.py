"""pybmt: biconvex minimum-time trajectory planning with Bezier curves.

Public surface (populated as components are ported):

    from pybmt import MinimumTimePlanner, solve_minimum_time
    from pybmt import PolygonalInitializer
    from pybmt import EISCSPlanner, solve_scs_minimum_time
    from pybmt import SolveResult, SCSResult
"""

from ._version import __version__
from .planner import MinimumTimePlanner, solve_minimum_time
from .polygonal import PolygonalInitializer
from .result import SCSResult, SolveResult
from .scs import EISCSPlanner, solve_scs_minimum_time

__all__ = [
    "__version__",
    "MinimumTimePlanner",
    "solve_minimum_time",
    "PolygonalInitializer",
    "EISCSPlanner",
    "solve_scs_minimum_time",
    "SolveResult",
    "SCSResult",
]
