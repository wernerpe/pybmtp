- [done] Handle jerk and snap with a unified UI: a single `Limits(velocity,
  acceleration, jerk=None, snap=None)` object threaded through every planner. The
  `T_powers` ladder is built only as deep as the highest supplied limit (J), so
  velocity+acceleration is byte-identical to the old acceleration-only program;
  `continuity_order`/`terminal_order` are configurable (and may not exceed J).
  Snap-constrained drone example shipped: `examples/village_flythrough.py`.

- [todo] Change the polygonal initializer to use the biconvex trajectory update
  with no planes (`TrajectoryUpdater.solve({})`) instead of the SCS 1D
  projection, reusing a single `TrajectoryUpdater` instance across segments (the
  trajectory degrees match). See `TODO(pete)` in `src/pybmt/polygonal.py`. This
  would also let the warm start honor jerk/snap, which the 1D projection can't.
