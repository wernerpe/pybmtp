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

The Bezier collision kernel is a small vendored C++ extension (Eigen + OpenMP),
so no GPU stack is required.

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
  10×10 workspace; a 7-waypoint seed path routes around them and the
  `MinimumTimePlanner` (BCP) refines it into a minimum-time trajectory under
  velocity/acceleration limits. Renders the seed and optimized trajectory to
  `examples/_out/guiding_example.png`.
- **`dual_arm_unload.py`** — a dual-arm pallet unload mirrored as a
  pure-geometry problem (no diff-IK, no URDF). Two arms each lift a brick off a
  shared pallet and carry it to a color-matched offload zone; the planner works
  directly on the 6D task-space state `[red_xyz, blue_xyz]` with
  configuration-space HPolyhedron collision obstacles, then publishes a meshcat
  animation of the bricks and grippers.
