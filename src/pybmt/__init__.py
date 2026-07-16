"""pybmt: biconvex minimum-time trajectory planning with Bezier curves.

Public surface (populated as components are ported):

    from pybmt import Limits                       # velocity/acceleration/jerk/snap sets
    from pybmt import MinimumTimePlanner, solve_minimum_time
    from pybmt import polygonal_initialization
    from pybmt import EISCSPlanner, solve_scs_minimum_time
    from pybmt import SolveResult, SCSResult
"""

from ._version import __version__
from .limits import Limits
from .planner import MinimumTimePlanner, solve_minimum_time
from .polygonal import polygonal_initialization
from .result import SCSResult, SolveResult
from .scs import EISCSPlanner, solve_scs_minimum_time

__all__ = [
    "__version__",
    "Limits",
    "MinimumTimePlanner",
    "solve_minimum_time",
    "polygonal_initialization",
    "EISCSPlanner",
    "solve_scs_minimum_time",
    "SolveResult",
    "SCSResult",
]
