#pragma once

#include <time.h>

#include <memory>
#include <set>
#include <mutex>
#include <random>
#include <atomic>

#include "args.h"
#include "dictionary.h"
#include "model.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace minkowski {

static const int32_t NEGATIVE_TABLE_SIZE = 100000000; // increased from the original

class Minkowski {
protected:
    std::shared_ptr<Args> args_;
    std::shared_ptr<Dictionary> dict_;

    std::shared_ptr<std::vector<Vector>> vectors_;
    std::shared_ptr<std::vector<std::mutex>> vector_flags_;

    std::shared_ptr<std::vector<int32_t>> negatives_;
    std::shared_ptr<Model> model_;
    std::atomic<bool> burnin_;

    void train_epochs(int32_t num_epochs, int32_t seed, real start_lr, real end_lr, bool checkpoint);

    void save_checkpoint(int32_t epochs_trained);

    /*
     * Given a vector of the word counts, generate a vector of negative samples
     * to be used.
     */
    void generate_negative_samples(const std::vector<int64_t>&);

    /*
     * Lock both the source and target; if this fails, return false; if it
     * succeeds, then proceed to lock the specified number of negative samples,
     * which are guaranteed to be distinct, and return true, in which case the
     * vector `samples` is populated with target, and then the negative samples.
     * If false is returned, then `samples` is unchanged.
     */
    bool obtain_vectors(int32_t source, int32_t target, std::vector<int32_t>& samples, int32_t num_negatives, std::minstd_rand& rng);

    /*
     * Release the locks of source and all the samples provided.
     */
    void release_vectors(int32_t source, std::vector<int32_t>& samples);

    /*
     * Return the word id of a negative sample, sampled uniformly at random
     * from the pregenerated list of negative samples using this->rng.
     * Guaranteed to not coincide with the provided index `target`.
     */
    int32_t get_negative_sample(int32_t target, std::minstd_rand& rng);

public:
    Minkowski(std::shared_ptr<Args> args);

    void save_vectors(std::string);
    void print_info(clock_t, real, int64_t, real, real);

    void skipgram(Model&, real, const std::vector<int32_t>&, std::minstd_rand& rng);
    void epoch_thread(int32_t thread_id, int32_t seed, real start_lr, real end_lr);
    void train();

};
}
