# PyBMT: Biconvex Minimum-time Trajectory Planning Around Convex Obstacles

[![CI](https://github.com/wernerpe/pybmt/actions/workflows/ci.yml/badge.svg)](https://github.com/wernerpe/pybmt/actions/workflows/ci.yml)
[![Lint](https://github.com/wernerpe/pybmt/actions/workflows/lint.yaml/badge.svg)](https://github.com/wernerpe/pybmt/actions/workflows/lint.yaml)

Software accompanying the paper. More details to follow.

`pybmt` plans time-optimal Bezier trajectories that respect velocity and
acceleration limits while avoiding convex obstacles. The software contains three planners:

- **`MinimumTimePlanner`** — the headline biconvex planner (BCP): an outer loop
  alternating a trajectory-update SOCP and a separating-plane-update LP, with a
  parallel Bezier collision check between them.
- **`PolygonalInitializer`** — a fast constraint-respecting polygonal warm start
  used to seed the biconvex planners.
- **`EISCSPlanner`** — Combination of Edge-Inflation for convex obstacles paired with SCSPlanning as used in the paper.

The Bezier collision check is a small C++ extension built on Eigen, with
optional OpenMP parallelism.

## Install

`pybmt` is not yet on PyPI; install it from a checkout of this repository:

```bash
pip install .
```

This builds the C++ collision kernel and pulls the runtime dependencies:
`drake` (pydrake) and `pybezier` from PyPI, plus `scsplanning` (pinned to a
specific upstream commit, since it has no PyPI release).

## Examples

Requires [`uv`](https://docs.astral.sh/uv/). The `examples/` directory is a uv
project, so `uv run` builds the environment (including the right Python) on first
run — no manual install needed:

```bash
cd examples
uv run box_obstacles_2d.py                          # 2D box-obstacle demo (BCP vs SCS)
uv run dual_arm_unload.py --planner bcp             # dual-arm pallet unload
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
cmake -S . -B build-cpp -DPYBMT_BUILD_TESTS=ON -DPYBMT_BUILD_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp --parallel
ctest --test-dir build-cpp --output-on-failure
```
