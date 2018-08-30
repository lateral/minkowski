#pragma once

#include <vector>
#include <utility>
#include <memory>
#include <mutex>

#include "args.h"
#include "vector.h"
#include "real.h"

namespace minkowski {

class Model {
protected:
    std::shared_ptr<std::vector<Vector>> vectors_;
    std::shared_ptr<Args> args_;
    std::shared_ptr<std::vector<std::mutex>> vector_flags_;
    Vector acc_grad_source_;
    Vector grad_output_;
    real performance_;
    int64_t nexamples_;
    real* t_sigmoid;

    void precompute_sigmoid();

public:
    Model(std::shared_ptr<std::vector<Vector>> vectors,
          std::shared_ptr<Args> args);
    ~Model();

    real binary_logistic(Vector& input, int32_t, bool, real);

    void log_bilinear_negative_sampling(int32_t source, std::vector<int32_t>& samples, real lr);

    /*
     * Return a metric on the average performance of this model since the last
     * call to this function (so this function is not idempotent).
     */
    real get_performance();

    real sigmoid(real) const;

    /*
     * Update (in place) the hyperboloid point in the direction of its
     * (hyperboloid-)tangent vector `tangent`.  Uses the exponential map
     * on the hyperboloid.
     */
    void update(Vector& point, Vector& tangent);
};

}
