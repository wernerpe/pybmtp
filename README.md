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

```bash
pip install pybmt
```

Requires `drake` (pydrake), `pybezier`, and `scsplanning`.

## Examples

See `examples/` for runnable `uv` scripts:

```bash
uv run examples/02_minimum_time_2d.py
```
