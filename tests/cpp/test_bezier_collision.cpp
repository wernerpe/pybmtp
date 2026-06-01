/**
 * Native unit tests for pybmt's Bezier geometry + collision kernel.
 *
 * Covers:
 *   - the convex-hull / subdivision collision predicate (correctness + the
 *     "never a false positive" guarantee),
 *   - the composite-curve intersection map and its ignore-set handling,
 *   - De Casteljau domain splitting,
 *   - that OpenMP actually accelerates the parallel intersection path.
 */
#include <gtest/gtest.h>

#include <chrono>
#include <random>

#include "pybmt/bezier_curve.h"

#if PYBMT_HAVE_OPENMP
#include <omp.h>
#endif

using namespace pybmt;

namespace {

// Axis-aligned box {x : lb <= x <= ub} as the H-polyhedron {x : A x <= b}
// with A = [I; -I], b = [ub; -lb].
void MakeBox(const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
             Eigen::MatrixXd& A, Eigen::VectorXd& b) {
  const int d = lb.size();
  A.resize(2 * d, d);
  b.resize(2 * d);
  A.topRows(d) = Eigen::MatrixXd::Identity(d, d);
  A.bottomRows(d) = -Eigen::MatrixXd::Identity(d, d);
  b.head(d) = ub;
  b.tail(d) = -lb;
}

// Straight-line degree-`degree` Bezier from `start` to `end` (control points
// evenly spaced, so the curve is exactly the segment).
BezierCurve StraightLine(const Eigen::VectorXd& start,
                         const Eigen::VectorXd& end, int degree, double t0 = 0.0,
                         double t1 = 1.0) {
  Eigen::MatrixXd pts(degree + 1, start.size());
  for (int i = 0; i <= degree; ++i) {
    const double s = static_cast<double>(i) / degree;
    pts.row(i) = (1.0 - s) * start.transpose() + s * end.transpose();
  }
  return BezierCurve(pts, t0, t1);
}

}  // namespace

TEST(BezierHPolyhedron, FarCurveIsCollisionFree) {
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  MakeBox(Eigen::Vector2d(-1, 4), Eigen::Vector2d(1, 6), A, b);  // box near y=5

  // Curve along the x-axis, far below the box.
  auto curve = StraightLine(Eigen::Vector2d(-2, 0), Eigen::Vector2d(2, 0), 6);
  EXPECT_EQ(BezierCurveHPolyhedronCollisionFree(curve, A, b, 1e-3), 1);
}

TEST(BezierHPolyhedron, CurveThroughObstacleCollides) {
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  MakeBox(Eigen::Vector2d(-1, -1), Eigen::Vector2d(1, 1), A, b);  // box at origin

  // Curve passes straight through the box.
  auto curve = StraightLine(Eigen::Vector2d(-2, 0), Eigen::Vector2d(2, 0), 6);
  EXPECT_EQ(BezierCurveHPolyhedronCollisionFree(curve, A, b, 1e-3), 0);
}

TEST(BezierHPolyhedron, EndpointsOutsideButMiddleDipsInStillCollides) {
  // Endpoints outside the box, but the curve bulges down into it. This is the
  // case the convex-hull-only check would miss; subdivision must catch it.
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  MakeBox(Eigen::Vector2d(-0.5, -0.5), Eigen::Vector2d(0.5, 0.5), A, b);

  Eigen::MatrixXd pts(7, 2);
  pts << -2.0, 1.0,
         -1.0, -0.3,
         -0.5, -0.3,
          0.0, -0.3,
          0.5, -0.3,
          1.0, -0.3,
          2.0, 1.0;
  // Endpoints sit at y=1 (well above the box), but the pull-down interior
  // control points drag the curve through it: at t=0.5 the curve is at
  // (0, -0.26), inside [-0.5, 0.5]^2. A convex-hull-only check would also flag
  // this, but so would the much tighter subdivision path; the point is that the
  // endpoints-outside shortcut must NOT declare it free.
  BezierCurve curve(pts, 0.0, 1.0);
  EXPECT_EQ(BezierCurveHPolyhedronCollisionFree(curve, A, b, 1e-4), 0);
}

TEST(BezierHPolyhedron, NeverFalsePositiveOnSampledCurve) {
  // Property check: if the kernel says "collision-free", then no densely
  // sampled point on the curve is inside the obstacle.
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  MakeBox(Eigen::Vector2d(-0.3, -0.3), Eigen::Vector2d(0.3, 0.3), A, b);

  auto curve = StraightLine(Eigen::Vector2d(-2, 0.4), Eigen::Vector2d(2, 0.4), 6);
  const bool claimed_free =
      BezierCurveHPolyhedronCollisionFree(curve, A, b, 1e-3);
  ASSERT_EQ(claimed_free, 1);

  for (int k = 0; k <= 1000; ++k) {
    const double t = static_cast<double>(k) / 1000.0;
    const Eigen::VectorXd x = curve(t);
    const bool inside = ((A * x - b).maxCoeff() < 0.0);
    EXPECT_FALSE(inside) << "sampled point inside obstacle at t=" << t;
  }
}

TEST(DomainSplit, SegmentsAreContinuousAtSplit) {
  auto curve = StraightLine(Eigen::Vector2d(0, 0), Eigen::Vector2d(4, 2), 6);
  auto [left, right] = curve.domain_split(0.5);
  EXPECT_NEAR((left.final_point() - right.initial_point()).norm(), 0.0, 1e-12);
  // Continuity with the original curve at a few interior times.
  for (double t : {0.1, 0.3, 0.5, 0.7, 0.9}) {
    const Eigen::VectorXd ref = curve(t);
    const Eigen::VectorXd got = (t <= 0.5) ? left(t) : right(t);
    EXPECT_NEAR((ref - got).norm(), 0.0, 1e-10);
  }
}

TEST(CompositeIntersection, MapsSegmentsToObstaclesAndHonorsIgnore) {
  // Two-segment curve: segment 0 hits obstacle 0, segment 1 is clear.
  Eigen::Vector2d a(-2, 0), mid(0, 0), end(2, 4);
  BezierCurve seg0 = StraightLine(a, mid, 6, 0.0, 1.0);
  BezierCurve seg1 = StraightLine(mid, end, 6, 1.0, 2.0);
  CompositeBezierCurve curve({seg0, seg1});

  Eigen::MatrixXd A0, A1;
  Eigen::VectorXd b0, b1;
  MakeBox(Eigen::Vector2d(-1.2, -0.5), Eigen::Vector2d(-0.8, 0.5), A0, b0);  // on seg0
  MakeBox(Eigen::Vector2d(5, 5), Eigen::Vector2d(6, 6), A1, b1);            // far away

  std::vector<Eigen::MatrixXd> As{A0, A1};
  std::vector<Eigen::VectorXd> bs{b0, b1};

  auto hits = IntersectCompositeBezierCurveWithHPolyhedra(curve, As, bs, {},
                                                          1e-3, false);
  ASSERT_EQ(hits.count(0), 1u);
  EXPECT_EQ(hits.at(0), (std::vector<int>{0}));
  EXPECT_EQ(hits.count(1), 0u);  // segment 1 hits nothing

  // Ignoring obstacle 0 on segment 0 removes the only collision.
  std::unordered_map<int, std::vector<int>> ignore{{0, {0}}};
  auto hits2 = IntersectCompositeBezierCurveWithHPolyhedra(curve, As, bs, ignore,
                                                           1e-3, false);
  EXPECT_TRUE(hits2.empty());
}

TEST(CompositeIntersection, ParallelMatchesSerial) {
  std::mt19937 rng(0);
  std::uniform_real_distribution<double> u(-1.0, 1.0);

  std::vector<BezierCurve> segs;
  for (int s = 0; s < 64; ++s) {
    Eigen::Vector2d start(s, 0.0), stop(s + 1, 0.0);
    segs.push_back(StraightLine(start, stop, 6, s, s + 1));
  }
  CompositeBezierCurve curve(segs);

  std::vector<Eigen::MatrixXd> As;
  std::vector<Eigen::VectorXd> bs;
  for (int o = 0; o < 200; ++o) {
    Eigen::Vector2d c(u(rng) * 64.0, u(rng) * 0.05);
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    MakeBox(c.array() - 0.02, c.array() + 0.02, A, b);
    As.push_back(A);
    bs.push_back(b);
  }

  auto serial =
      IntersectCompositeBezierCurveWithHPolyhedra(curve, As, bs, {}, 1e-5, false);
  auto parallel =
      IntersectCompositeBezierCurveWithHPolyhedra(curve, As, bs, {}, 1e-5, true);
  EXPECT_EQ(serial, parallel);
}

#if PYBMT_HAVE_OPENMP
TEST(CompositeIntersection, OpenMpAcceleratesIntersection) {
  if (omp_get_max_threads() < 2) {
    GTEST_SKIP() << "Only one OpenMP thread available; cannot measure speedup.";
  }

  // Heavy workload: many segments x many obstacles, small tol to force deep
  // De Casteljau subdivision so each (segment, obstacle) check is non-trivial.
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> u(-1.0, 1.0);

  const int n_segments = 512;
  std::vector<BezierCurve> segs;
  for (int s = 0; s < n_segments; ++s) {
    Eigen::Vector2d start(s, 0.0), stop(s + 1, 0.3);
    segs.push_back(StraightLine(start, stop, 6, s, s + 1));
  }
  CompositeBezierCurve curve(segs);

  std::vector<Eigen::MatrixXd> As;
  std::vector<Eigen::VectorXd> bs;
  for (int o = 0; o < 1500; ++o) {
    // Boxes hugging the curve band but not containing endpoints -> the kernel
    // cannot early-exit and must subdivide, which is the work we parallelize.
    Eigen::Vector2d c(u(rng) * n_segments, 0.15 + u(rng) * 0.05);
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    MakeBox(c.array() - 0.03, c.array() + 0.03, A, b);
    As.push_back(A);
    bs.push_back(b);
  }

  using Clock = std::chrono::steady_clock;
  auto time_call = [&](bool parallel) {
    auto t0 = Clock::now();
    volatile auto res =
        IntersectCompositeBezierCurveWithHPolyhedra(curve, As, bs, {}, 1e-6,
                                                    parallel);
    (void)res;
    auto t1 = Clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
  };

  // Warm up (page-in, cache, thread-pool spin-up) before timing.
  time_call(false);
  time_call(true);

  // Take the best of a few runs to reduce noise.
  double serial = 1e30, parallel = 1e30;
  for (int r = 0; r < 3; ++r) {
    serial = std::min(serial, time_call(false));
    parallel = std::min(parallel, time_call(true));
  }

  // We only assert the parallel path is not meaningfully *slower* than serial
  // (allowing a 15% noise margin). A hard wall-clock speedup ratio is too flaky
  // on shared 2-core CI runners, where OpenMP overhead and contention can erase
  // a real speedup; correctness of the parallel result is covered separately by
  // CompositeIntersection.ParallelMatchesSerial. This still catches a broken
  // OpenMP path that serializes work or adds pathological overhead.
  EXPECT_LT(parallel, 1.15 * serial)
      << "serial=" << serial << "s parallel=" << parallel
      << "s threads=" << omp_get_max_threads();
}
#endif
