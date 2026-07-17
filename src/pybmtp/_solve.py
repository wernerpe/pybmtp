"""Parallel/serial solve shim for independent MathematicalPrograms.

``pydrake.all.SolveInParallel`` is only available on newer drake wheels. Notably
the last drake wheel built for macOS 13 (1.33.0) does not expose it, so importing
it unconditionally would force a drake floor that has no macOS-13 wheel. The
programs we batch are independent, so solving them serially is semantically
identical (just slower); we use ``SolveInParallel`` when present and fall back to
a serial loop otherwise.
"""

from __future__ import annotations

from typing import List, Optional

import pydrake.all as pd


def solve_in_parallel(
    progs: List[pd.MathematicalProgram],
    solver_options: Optional[pd.SolverOptions] = None,
) -> List[pd.MathematicalProgramResult]:
    """Solve independent programs, in parallel when drake supports it."""
    if hasattr(pd, "SolveInParallel"):
        return pd.SolveInParallel(
            progs=progs,
            solver_options=solver_options,
            parallelism=pd.Parallelism.Max(),
        )
    return [pd.Solve(prog, None, solver_options) for prog in progs]
