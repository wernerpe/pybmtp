"""pybmtp: biconvex minimum-time trajectory planning with Bezier curves.

Public surface:

    from pybmtp import Limits                       # velocity/acceleration/jerk/snap sets
    from pybmtp import MinimumTimePlanner, solve_minimum_time
    from pybmtp import polygonal_initialization
    from pybmtp import solve_in_parallel            # parallel Drake solve helper
    from pybmtp import SolveResult
"""

from ._solve import solve_in_parallel
from ._version import __version__
from .limits import Limits
from .planner import MinimumTimePlanner, solve_minimum_time
from .polygonal import polygonal_initialization
from .result import SolveResult

__all__ = [
    "__version__",
    "Limits",
    "MinimumTimePlanner",
    "solve_minimum_time",
    "polygonal_initialization",
    "solve_in_parallel",
    "SolveResult",
]
