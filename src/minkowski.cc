#include "minkowski.h"

#include <math.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <random>

// how many tokens to process before reporting on performance
constexpr int32_t REPORTING_INTERVAL = 50;

namespace minkowski {

Minkowski::Minkowski(std::shared_ptr<Args> args) {
    burnin_ = false;
    args_ = args;
}

void Minkowski::save_vectors(std::string fn) {
    std::ofstream ofs(fn + ".csv");
    if (!ofs.is_open()) {
        throw std::invalid_argument(fn + " cannot be opened for saving vectors!");
    }
    Vector vec(args_->dimension);
    for (int32_t i = 0; i < dict_->nwords_; i++) {
        std::string word = dict_->words_[i].word;
        vec = vectors_->at(i);
        ofs << word << " " << vec << std::endl;
    }
    ofs.close();
}

void Minkowski::print_info(clock_t start, real progress, int64_t tokens_processed, real lr, real performance) {
    real cpu_time_single_thread = real(clock() - start) / (CLOCKS_PER_SEC * args_->threads);
    real wst = real(tokens_processed) / cpu_time_single_thread;
    std::cerr << std::fixed;
    std::cerr << "\rProgress: " << std::setw(5) << std::setprecision(1) << 100 * progress << "%";
    std::cerr << "  words/sec/thread: " << std::setw(8) << std::setprecision(0) << wst;
    std::cerr << "  lr: " << std::setw(8) << std::setprecision(6) << lr;
    std::cerr << "  objective: " << std::setw(8) << std::setprecision(6) << performance;
    std::cerr << std::flush;
}

void Minkowski::skipgram(Model& model, real lr, const std::vector<int32_t>& line, std::minstd_rand& rng) {
    std::vector<int32_t> samples;
    int32_t num_negatives = args_->number_negatives;
    if (burnin_) {
        num_negatives /= 10;  // as per N&K
    }
    for (int32_t w = 0; w < line.size(); w++) {
        for (int32_t c = -args_->window_size; c <= args_->window_size; c++) {
            if (c != 0 && w + c >= 0 && w + c < line.size()) {
                int32_t source = line[w];
                int32_t target = line[w + c];
                if (!obtain_vectors(source, target, samples, num_negatives, rng)) {
                    // couldn't obtain one of the necessary locks, so skip!
                    continue;
                }
                model.log_bilinear_negative_sampling(source, samples, lr);
                release_vectors(source, samples);
            }
        }
    }
}

bool Minkowski::obtain_vectors(int32_t source, int32_t target, std::vector<int32_t>& samples, int32_t num_negatives, std::minstd_rand& rng) {
    if (!vector_flags_->at(source).try_lock()) {
        return false;
    }
    if (!vector_flags_->at(target).try_lock()) {
        vector_flags_->at(source).unlock();
        return false;
    }
    samples.clear();
    samples.push_back(target);

    while (samples.size() < num_negatives + 1) {
        auto next_negative = get_negative_sample(target, rng);
        if (vector_flags_->at(next_negative).try_lock()) {
            samples.push_back(next_negative);
        }
    }
    return true;
}

int32_t Minkowski::get_negative_sample(int32_t target, std::minstd_rand& rng) {
    int32_t negative;
    do {
        negative = negatives_->at(rng() % negatives_->size());
    } while (target == negative);
    return negative;
}

void Minkowski::release_vectors(int32_t source, std::vector<int32_t>& samples) {
    for (int32_t n = 0; n < samples.size(); n++) {
        vector_flags_->at(samples[n]).unlock();
    }
    vector_flags_->at(source).unlock();
}

void Minkowski::epoch_thread(int32_t thread_id, int32_t seed, real start_lr, real end_lr) {
    std::minstd_rand rng(seed);
    std::ifstream ifs(args_->input);
    utils::seek(ifs, thread_id * utils::size(ifs) / args_->threads);
    Model model(vectors_, args_);

    // number of tokens that this thread should process
    const int64_t max_tokens = dict_->ntokens_ / args_->threads;
    int64_t token_count = 0; // number processed so far
    int64_t iter_count = 0;
    std::vector<int32_t> line;
    clock_t start = clock();
    real lr = start_lr;
    real progress = 0.;
    while (token_count < max_tokens) {
        token_count += dict_->get_line(ifs, line, rng);
        progress = std::min(1.0, real(token_count) / max_tokens);
        lr = start_lr * (1.0 - progress) + end_lr * progress;
        skipgram(model, lr, line, rng);
        if (thread_id == 0) {
            // only thread 0 is responsible for printing progress info
            if (iter_count % REPORTING_INTERVAL == 0) {
                print_info(start, progress, token_count, lr, model.get_performance());
            }
        }
        iter_count++;
    }
    if (thread_id == 0) {
        print_info(start, progress, token_count, lr, model.get_performance());
        std::cerr << std::endl;
    }
    ifs.close();
}

void Minkowski::train() {
    std::ifstream ifs(args_->input);
    if (!ifs.is_open()) {
        throw std::invalid_argument(
                    args_->input + " cannot be opened for training!");
    }
    dict_ = std::make_shared<Dictionary>(args_);
    dict_->determine_vocabulary(ifs);
    ifs.close();
    // generate the negative samples
    negatives_ = std::make_shared<std::vector<int32_t>>();
    generate_negative_samples(dict_->get_counts());
    // initialise the vectors
    std::minstd_rand rng(args_->seed);
    Vector init_vector(args_->dimension);
    vectors_ = std::make_shared<std::vector<Vector>>();
    for (int64_t i=0; i < dict_->nwords_; i++) {
        random_hyperboloid_point(init_vector, rng, args_->init_std_dev);
        vectors_->push_back(init_vector);
    }
    vector_flags_ = std::shared_ptr<std::vector<std::mutex>>(new std::vector<std::mutex>(vectors_->size()));
    // do any burn-in epochs
    burnin_ = true;
    train_epochs(args_->burnin_epochs, args_->seed, args_->burnin_lr, args_->burnin_lr, false);
    burnin_ = false;
    // do the epochs: use a different seed to ensure different negative samples
    train_epochs(args_->epochs, -1 * (args_->seed), args_->start_lr, args_->end_lr, true);
}

void Minkowski::save_checkpoint(int32_t epochs_trained) {
    if (args_->checkpoint_interval > 0 && epochs_trained % args_->checkpoint_interval == 0) {
        // checkpoint (save) the vectors - pad epoch number to maintain
        // alphabetical ordering
        std::string epochs_done = std::to_string(epochs_trained);
        epochs_done = std::string(6 - epochs_done.length(), '0') + epochs_done;
        this->save_vectors(args_->output + "-after-" + epochs_done + "-epochs");
    }
}

void Minkowski::train_epochs(int32_t num_epochs, int32_t seed, real start_lr, real end_lr, bool checkpoint) {
    real lr_delta_per_epoch = (start_lr - end_lr) / num_epochs;
    for (int32_t epoch = 0; epoch < num_epochs; epoch++) {
        if (checkpoint) {
            save_checkpoint(epoch);
        }
        std::cerr << "\rEpoch: " << (epoch + 1) << " / " << num_epochs << "\n";
        std::cerr << std::flush;
        real epoch_start_lr = start_lr - real(epoch) * lr_delta_per_epoch;
        real epoch_end_lr = start_lr - real(epoch + 1) * lr_delta_per_epoch;
        std::vector<std::thread> threads;
        for (int32_t thread_id = 0; thread_id < args_->threads; thread_id++) {
            int32_t thread_seed = seed + epoch * args_->threads + thread_id;
            threads.push_back(std::thread([=]() {
                epoch_thread(thread_id, thread_seed, epoch_start_lr, epoch_end_lr);
            }));
        }
        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }
    }
    if (checkpoint) {
        save_checkpoint(num_epochs);
    }
}

void Minkowski::generate_negative_samples(const std::vector<int64_t>& counts) {
    real z = 0.0;
    for (size_t i = 0; i < counts.size(); i++) {
        z += pow(counts[i], args_->distribution_power);
    }
    for (size_t i = 0; i < counts.size(); i++) {
        real c = pow(counts[i], args_->distribution_power);
        for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
            negatives_->push_back(i);
        }
    }
}

}
