#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace minkowski {

class Args {
public:
    Args();
    std::string input;
    std::string output;
    double start_lr;
    double end_lr;
    double burnin_lr;
    double max_step_size;
    int seed;
    int dimension;
    int checkpoint_interval;
    double distribution_power;
    int window_size;
    int epochs;
    int burnin_epochs;
    int min_count;
    int number_negatives;
    int threads;
    double t;
    double init_std_dev;

    void parse_args(const std::vector<std::string>& args);
    void print_help();
};
}
