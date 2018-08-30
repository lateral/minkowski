#pragma once

#include <cstdint>
#include <ostream>
#include <random>
#include <assert.h>

#include "real.h"

namespace minkowski {

/*
 * Represent vectors in Minkowski space, where the last co-ordinate is
 * considered to be time-like.
 */
class Vector {

public:
    int64_t dimension_;
    real* data_;

    explicit Vector(int64_t);
    explicit Vector(const Vector&);
    ~Vector();

    Vector& operator= (const Vector&);
    real& operator[](int64_t);
    const real& operator[](int64_t) const;

    /*
     * Return the length of this vector.
     */
    int64_t size() const;

    /*
     * Set all entries to zero.
     */
    void zero();

    /*
     * Multiply all entries by the given value, in place.
     */
    void multiply(real);

    /*
     * Add the given vector to this vector.
     */
    void add(const Vector& source);

    /*
     * Add the specified multiple of the given vector to this vector.
     */
    void add(const Vector& other_vector, real scalar);

    /*
     * Calculate (in place) the projection of this hyperboloid point to the
     * Poincare ball.
     */
    void to_ball_point();

    /*
     * Calculate (in place) the point on the hyperboloid whose projection is
     * this poincare ball point.
     */
    void to_hyperboloid_point();

    /*
     * Calculate (in place) the Poincare ball tangent corresponding to this
     * vector, when interpreted as a hyperboloid tangent vector at the provided
     * point.
     */
    void to_ball_tangent(const Vector& hyperboloid_point);

    /*
     * Project this vector onto the hyperboloid tangent space at specified point.
     */
    void project_onto_tangent_space(const Vector& hyperboloid_point);

    /*
     * Replace this point (in place) with the point obtained by following the
     * geodesic in the direction of `tangent_unit_vec` for distance
     * `step_size`.
     * Pre: `tangent_unit_vec` is a unit vector; `step_size` > 0.
     */
    void geodesic_update(const Vector& tangent_unit_vec, real step_size);

    /*
     * Ensure that this time-like point is on the hyperboloid by
     * projecting it back, if necessary.  Used to ensure numerical stability.
     */
    void ensure_on_hyperboloid();

};

std::ostream& operator<<(std::ostream&, const Vector&);

/*
 * Return the Minkowski inner product of the two vectors provided, where the
 * last co-ordinate is interpreted as being time-like.
 */
inline real minkowski_dot(const Vector& v, const Vector& w) {
    real result = 0;
    int64_t n = v.size();

    for (int64_t i = 0; i < n-1; ++i) {
        result += v[i]*w[i];
    }
    result -= v[n-1]*w[n-1];
    return result;
}

/*
 * Sample from points on the hyperboloid distributed circularly
 * around the base point with the hyperbolic distance from the base
 * point normally distributed with standard deviation std_dev.
 */
void random_hyperboloid_point(Vector& vector, std::minstd_rand& rng, real std_dev);

/*
 * Return the distance between the two points on the hyperboloid.
 */
real distance(const Vector& point0, const Vector& point1);

/*
 * Return the gradient of the distance.
 * Gradient is in the ambient Minkowski space (so needs to be projected
 * onto the tangent plane).
 */
void distance_gradient(const Vector& varying_pt, const Vector& fixed_pt, Vector& gradient);

}
