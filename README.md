# PyBMTP: Biconvex Minimum-time Trajectory Planning Around Convex Obstacles

[![CI](https://github.com/wernerpe/pybmtp/actions/workflows/ci.yml/badge.svg)](https://github.com/wernerpe/pybmtp/actions/workflows/ci.yml)
[![Lint](https://github.com/wernerpe/pybmtp/actions/workflows/lint.yaml/badge.svg)](https://github.com/wernerpe/pybmtp/actions/workflows/lint.yaml)

Software accompanying the paper. More details to follow.

`pybmtp` plans time-optimal Bezier trajectories that respect a velocity limit and,
optionally, acceleration / jerk / snap limits while avoiding convex obstacles. The
package provides:

- **`MinimumTimePlanner`** — the headline biconvex planner (BMTP): an outer loop
  alternating a trajectory-update SOCP and a separating-plane-update LP, with a
  parallel Bezier collision check between them. Limits up to snap (4th
  derivative) and the junction-continuity order are configurable.
- **`polygonal_initialization`** — a constraint-respecting polygonal warm start
  (straight, rest-to-rest, minimum-time segments under the same limits) used to
  seed the biconvex planner.

The paper's **SCS + Edge-Inflation baseline** (`EISCSPlanner`) is intentionally not
part of the installable package: it depends on
[`scsplanning`](https://github.com/TobiaMarcucci/scsplanning), which is not on
PyPI. It ships with the experiments in [`examples/eiscs`](examples/eiscs) and is
run by the example scripts below.

The Bezier collision check is a small C++ extension built on Eigen, with
optional OpenMP parallelism.

## Install

`pybmtp` is not yet on PyPI; install it from a checkout of this repository:

```bash
pip install .
```

This builds the C++ collision kernel and pulls its runtime dependencies —
`drake` (pydrake) and `pybezier` — both from PyPI.

## Usage

Limits are bundled in a single `Limits` object. Only `velocity` is required
(without it the minimum time is unbounded); `acceleration`, `jerk` and `snap` are
optional and contiguous from velocity up. The program is built no larger than the
highest limit you give.

```python
import numpy as np, pydrake.all as pd
from pybmtp import Limits, MinimumTimePlanner

ball = lambda r: pd.Hyperellipsoid.MakeHypersphere(r, np.zeros(3))
limits = Limits(velocity=ball(5.0), acceleration=ball(5.0),
                jerk=ball(25.0), snap=ball(50.0))       # only velocity is required

result = MinimumTimePlanner(
    trajectory_degree=8,
    continuity_order=4,     # C1..C4 continuous (defaults to the limit order)
    terminal_order=2,       # rest velocity + acceleration at both ends
).solve(path, obstacles, domain, limits)

print(result.total_time, result.trajectory)
```

Drop `acceleration`/`jerk`/`snap` for a lower-order problem; `continuity_order` and
`terminal_order` default to the number of supplied limits and may not exceed it.

## Examples

Requires [`uv`](https://docs.astral.sh/uv/). The `examples/` directory is a uv
project, so `uv run` builds the environment (including the right Python) on first
run — no manual install needed:

```bash
cd examples
uv run box_obstacles_2d.py                          # 2D box-obstacle demo (BMTP vs SCS)
uv run dual_arm_unload.py --planner bmtp             # dual-arm pallet unload
uv run village_flythrough.py --no-meshcat           # snap-constrained UAV flight across the village
```

## Tests

The suite has two parts, mirroring CI.

**Python tests** (planners, collision wrapper, examples). Install the test
extras and run `pytest`:

```bash
pip install -e '.[test]'
pytest -q
```

**Native C++ tests** (the Bezier collision kernel, via gtest/ctest). Configure
a test build, compile it, and run `ctest`:

```bash
cmake -S . -B build-cpp -DPYBMTP_BUILD_TESTS=ON -DPYBMTP_BUILD_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp --parallel
ctest --test-dir build-cpp --output-on-failure
```
