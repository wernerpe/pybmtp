# pybmt

Biconvex minimum-time trajectory planning with Bezier curves.

`pybmt` plans time-optimal Bezier trajectories that respect velocity and
acceleration limits while avoiding convex obstacles. It ships three planners:

- **`MinimumTimePlanner`** — the headline biconvex planner (BCP): an outer loop
  alternating a trajectory-update SOCP and a separating-plane-update LP, with a
  parallel Bezier collision check between them.
- **`PolygonalInitializer`** — a fast constraint-respecting polygonal warm start
  used to seed the biconvex planners.
- **`SCSPlanner`** — the sequential-convex-shrinking variant using exact convex
  obstacle separation.

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

The scripts in `examples/` are self-contained [PEP 723](https://peps.python.org/pep-0723/)
scripts: [`uv`](https://docs.astral.sh/uv/) reads the inline dependency block at
the top and runs them against the local checkout, so no manual install is
needed:

```bash
uv run examples/guiding_example.py
uv run examples/dual_arm_unload.py
```

If you already have `pybmt` installed in your environment, run them with plain
Python instead:

```bash
python examples/guiding_example.py
```

- **`guiding_example.py`** — the headline 2D demo. Five box obstacles sit in a
  10×10 workspace; a 7-waypoint seed path routes around them. It solves the
  scene with both planners — the biconvex `MinimumTimePlanner` (BCP) and the
  `SCSPlanner` convex-region baseline (SCS) — and overlays both optimized
  trajectories against the seed on one legended figure,
  `examples/_out/guiding_example.png`.
- **`dual_arm_unload.py`** — a dual-arm pallet unload as a pure-geometry
  task-space problem. Two arms each lift a brick off a shared pallet and carry
  it to a color-matched offload zone. The planner works directly on the 6D
  task-space state `[red_xyz, blue_xyz]`, with each collision pair (brick vs
  brick, gripper vs brick, gripper vs gripper) expressed as a configuration-space
  HPolyhedron. It samples a random unloadable pallet, plans the simultaneous
  two-arm move, publishes a meshcat animation, and writes a 3D plot of each
  arm's waypoint seed vs. optimized path to
  `examples/_out/dual_arm_unload_<planner>.png`. Pick the planner with
  `--planner {bcp,scs}` (default `bcp`) and the pallet with `--seed`:

```bash
uv run examples/dual_arm_unload.py --planner scs --seed 3
```
