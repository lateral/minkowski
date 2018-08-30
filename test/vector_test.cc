#include "gtest/gtest.h"
#include "vector.h"
#include "real.h"
#include <cmath>
#include <random>

namespace {

TEST(VectorTest, init_with_zeros) {
    int m = 5;
    minkowski::Vector vec(m);
    vec.zero();
    EXPECT_EQ(vec.dimension_, m);
    for (auto i = 0; i < vec.dimension_; ++i) {
        EXPECT_EQ(0., vec[i]);
    }
}

TEST(VectorTest, multiply) {
    minkowski::Vector vec(2);
    vec[0] = 1.;
    vec[1] = 2.;
    vec.multiply(1.5);
    EXPECT_FLOAT_EQ(1.5, vec[0]);
    EXPECT_FLOAT_EQ(3., vec[1]);
}

TEST(VectorTest, minkowskiDot) {
    minkowski::Vector vec_a(3);
    minkowski::Vector vec_b(3);

    vec_a[0] = 1.;
    vec_a[1] = 0.5;
    vec_a[2] = -2.;

    vec_b[0] = 0.;
    vec_b[1] = 0.5;
    vec_b[2] = 1.;

    auto mdp = minkowski_dot(vec_a, vec_b);
    EXPECT_FLOAT_EQ(2.25, mdp);
}

TEST(VectorTest, randomHyperboloidPoint) {
    std::minstd_rand rng(1);
    minkowski::Vector vec_a(3);
    minkowski::Vector vec_b(3);

    random_hyperboloid_point(vec_a, rng, 0.1);
    random_hyperboloid_point(vec_b, rng, 0.1);
    EXPECT_NE(vec_a[0], vec_b[0]);  // vectors should be different

    // vectors should be on the hyperboloid
    auto mdp = minkowski_dot(vec_a, vec_a);
    EXPECT_FLOAT_EQ(-1., mdp);
}

TEST(VectorTest, distance) {
    minkowski::Vector vec_a(2);
    minkowski::Vector vec_b(2);

    // basepoint
    vec_a[0] = 0.;
    vec_a[1] = 1.0;

    real hyperangle = 0.5;
    vec_b[0] = std::sinh(hyperangle);
    vec_b[1] = std::cosh(hyperangle);

    real dist = distance(vec_a, vec_b);
    EXPECT_FLOAT_EQ(hyperangle, dist);
}

TEST(VectorTest, ensureOnHyperboloid) {
    minkowski::Vector vec(2);

    // almost the basepoint
    vec[0] = 0.;
    vec[1] = 1.000001;

    // should now be the basepoint
    vec.ensure_on_hyperboloid();
    EXPECT_FLOAT_EQ(0.0, vec[0]);
    EXPECT_FLOAT_EQ(1.0, vec[1]);
}

TEST(VectorTest, ensureOnHyperboloidNoOp) {
    minkowski::Vector vec(2);

    // basepoint: already on the hyperboloid
    vec[0] = 0.;
    vec[1] = 1.0;
    vec.ensure_on_hyperboloid();
    // nothing should have changed
    EXPECT_FLOAT_EQ(0., vec[0]);
    EXPECT_FLOAT_EQ(1., vec[1]);
}

TEST(VectorTest, toBallPointAtBasepoint) {
    minkowski::Vector vec(2);
    // basepoint
    vec[0] = 0.;
    vec[1] = 1.0;
    vec.to_ball_point();
    // should be centre of Poincaré disc
    EXPECT_FLOAT_EQ(0., vec[0]);
    EXPECT_FLOAT_EQ(0., vec[1]);
}

TEST(VectorTest, toBallPoint) {
    minkowski::Vector vec(2);
    real dist = 1;
    vec[0] = std::sinh(dist);
    vec[1] = std::cosh(dist);

    vec.to_ball_point();
    real norm = std::sqrt(minkowski_dot(vec, vec));
    EXPECT_FLOAT_EQ(std::tanh(dist / 2), norm);
}

TEST(VectorTest, toHyperboloidPoint) {
    minkowski::Vector vec(3);
    real dist = 1.2;
    vec[0] = 0.;
    vec[1] = std::tanh(dist / 2);
    vec[2] = 0;
    vec.to_hyperboloid_point();
    EXPECT_FLOAT_EQ(0., vec[0]);
    EXPECT_FLOAT_EQ(std::sinh(dist), vec[1]);
    EXPECT_FLOAT_EQ(std::cosh(dist), vec[2]);
}

TEST(VectorTest, toBallTangent) {
    // a point on the hyperboloid
    minkowski::Vector point(3);
    real dist = 1.2;
    point[0] = std::sinh(dist);
    point[1] = 0.;
    point[2] = std::cosh(dist);

    // a unit tangent vector in its tangent space
    minkowski::Vector tangent(3);
    tangent[0] = 0.;
    tangent[1] = 1.;
    tangent[2] = 0.;

    // map to the corresponding tangent vector in the tangent space of the ball point
    tangent.to_ball_tangent(point);

    // check some obvious co-ordinates
    EXPECT_FLOAT_EQ(0., tangent[0]); // since hasn't changed rotational angle
    EXPECT_FLOAT_EQ(0., tangent[2]); // since it is tangent to the poincare disc

    // check its length
    real r = std::tanh(dist / 2); // displacement of corres. ball point from origin
    real euclid_norm = std::sqrt(minkowski_dot(tangent, tangent));
    // Euclidean norm is related to the tangent norm via r
    // Should be 1., since it is a unit vector in the tangent space of the
    // poincare disc (since it was a unit vector in the tangent space of the
    // hyperboloid, and the metric on the disc is induced).
    EXPECT_FLOAT_EQ(1., 2 * euclid_norm / (1 - r * r));
}

TEST(VectorTest, geodesicUpdate) {
    // basepoint
    minkowski::Vector basepoint(2);
    basepoint[0] = 0.f;
    basepoint[1] = 1.0f;
    // our test point: start out at the basepoint
    minkowski::Vector point(basepoint);
    // a tangent vector in its tangent space
    real dist = 3;
    minkowski::Vector tangent(2);
    tangent[0] = 1;
    tangent[1] = 0.;
    // apply exponential
    point.geodesic_update(tangent, dist);
    // exponential map is a radial isometry, so should be dist from basepoint
    EXPECT_FLOAT_EQ(dist, distance(basepoint, point));
}

TEST(VectorTest, projectOntoTangentSpace) {
    // basepoint
    minkowski::Vector point(2);
    point[0] = 0.;
    point[1] = 1.0;
    minkowski::Vector tangent(2);
    tangent[0] = 1.5;
    tangent[1] = 1.0;
    tangent.project_onto_tangent_space(point);
    // tangent should be orthogonal to the vector of the point
    real mdp = minkowski_dot(tangent, point);
    EXPECT_FLOAT_EQ(0., mdp);
}

}  // namespace
