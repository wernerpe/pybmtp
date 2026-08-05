# PyBMTP: Biconvex Minimum-time Trajectory Planning Around Convex Obstacles

[![arXiv](https://img.shields.io/badge/arXiv-2608.02834-b31b1b.svg)](https://arxiv.org/abs/2608.02834)
[![Website](https://img.shields.io/badge/website-bmtp-blue.svg)](https://wernerpe.github.io/bmtp-website/)
[![CI](https://github.com/wernerpe/pybmtp/actions/workflows/ci.yml/badge.svg)](https://github.com/wernerpe/pybmtp/actions/workflows/ci.yml)
[![Lint](https://github.com/wernerpe/pybmtp/actions/workflows/lint.yaml/badge.svg)](https://github.com/wernerpe/pybmtp/actions/workflows/lint.yaml)

Software accompanying [arXiv:2608.02834](https://arxiv.org/abs/2608.02834).

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
- **[`examples/`](examples)** — runnable demos, including the SCS + Edge-Inflation
  baseline (`EISCSPlanner`) from the paper.

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

**Python tests** (the BMTP planner, polygonal warm start, and collision wrapper).
Install the test extras and run `pytest`:

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

## Citation

If you use this software in your research, please cite the accompanying paper:

```bibtex
@article{werner2026biconvex,
  author        = {Werner, Peter and Marcucci, Tobia and Rus, Daniela},
  title         = {Biconvex Optimization for Smooth Minimum-Time Trajectories around Convex Obstacles},
  journal       = {arXiv preprint arXiv:2608.02834},
  year          = {2026},
  eprint        = {2608.02834},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
}
```

## Funding

We are very grateful for the funding provided by the Office of Naval Research,
Award Number N00014-23-1-2354. Research was also sponsored in part by the
Department of the Air Force Artificial Intelligence Accelerator and was
accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and
conclusions contained in this document are those of the authors and should not
be interpreted as representing the official policies, either expressed or
implied, of the Department of the Air Force or the U.S. Government. The U.S.
Government is authorized to reproduce and distribute reprints for Government
purposes notwithstanding any copyright notation herein.
