#include "model.h"

#include <iostream>
#include <assert.h>
#include <algorithm>
#include <thread>

namespace minkowski {

constexpr int32_t SIGMOID_TABLE_SIZE = 512;
constexpr int32_t MAX_SIGMOID = 8;
constexpr real MIN_STEP_SIZE = 1e-10;
constexpr real SHIFT = 3.0;
constexpr real MAX_MINKOWSKI_DOT = -1 - 1e-10;

Model::Model(std::shared_ptr<std::vector<Vector>> vectors,
             std::shared_ptr<Args> args)
    : acc_grad_source_(args->dimension),
      grad_output_(args->dimension) {
    vectors_ = vectors;
    args_ = args;
    performance_ = 0.0;
    nexamples_ = 1;
    precompute_sigmoid();
}

Model::~Model() {
    delete[] t_sigmoid;
}

real Model::binary_logistic(Vector& input, int32_t target, bool label, real lr) {
    real score = sigmoid(minkowski_dot(input, vectors_->at(target)) + SHIFT);
    real delta = real(label) - score;

    // accumulate the unprojected gradient for the input word vector
    acc_grad_source_.add(vectors_->at(target), delta);

    // update the output word vector
    grad_output_ = input;
    grad_output_.multiply(lr * delta);
    grad_output_.project_onto_tangent_space(vectors_->at(target));
    update(vectors_->at(target), grad_output_);

    if (label) {
        return -std::log(score + 1e-8);
    } else {
        return -std::log(1.0 - score + 1e-8);
    }
}

void Model::update(Vector& point, Vector& tangent) {
    real step_size = std::sqrt(minkowski_dot(tangent, tangent));
    // normalize the tangent vector
    tangent.multiply(1.0 / step_size);
    if (step_size < MIN_STEP_SIZE) {
        return;
    }
    // clip the step size, if needed
    if (step_size > args_->max_step_size) {
        step_size = args_->max_step_size;
    }
    // geodesic update
    point.geodesic_update(tangent, step_size);
}

void Model::log_bilinear_negative_sampling(int32_t source, std::vector<int32_t>& samples, real lr) {
    acc_grad_source_.zero();
    for (int32_t n = 0; n < samples.size(); n++) {
        performance_ += binary_logistic(vectors_->at(source), samples[n], n == 0, lr);
    }
    nexamples_ += 1;

    acc_grad_source_.multiply(lr);
    acc_grad_source_.project_onto_tangent_space(vectors_->at(source));
    update(vectors_->at(source), acc_grad_source_);
}

real Model::get_performance() {
    real avg = performance_ / nexamples_;
    performance_ = 0.0;
    nexamples_ = 1;
    return avg;
}

void Model::precompute_sigmoid() {
    t_sigmoid = new real[SIGMOID_TABLE_SIZE + 1];
    for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
        real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
        t_sigmoid[i] = 1.0 / (1.0 + std::exp(-x));
    }
}

real Model::sigmoid(real x) const {
    if (x < -MAX_SIGMOID) {
        return 0.0;
    } else if (x > MAX_SIGMOID) {
        return 1.0;
    } else {
        int i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
        return t_sigmoid[i];
    }
}

}
