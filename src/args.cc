#include "args.h"

#include <stdlib.h>

#include <iostream>
#include <stdexcept>

namespace minkowski {

Args::Args() {
    start_lr = 0.05;
    end_lr = 0.05;
    burnin_lr = 0.05;
    max_step_size = 2.0;
    dimension = 100;
    window_size = 5;
    checkpoint_interval = -1;
    distribution_power = 0.5;
    epochs = 5;
    burnin_epochs = 0;
    min_count = 5;
    number_negatives = 5;
    threads = 12;
    t = 1e-4;
    init_std_dev = 0.1;
    seed = 1;
}


void Args::parse_args(const std::vector<std::string>& args) {
    for (int ai = 1; ai < args.size(); ai += 2) {
        if (args[ai][0] != '-') {
            std::cerr << "Provided argument without a dash! Usage:" << std::endl;
            print_help();
            exit(EXIT_FAILURE);
        }
        try {
            if (args[ai] == "-h") {
                std::cerr << "Here is the help! Usage:" << std::endl;
                print_help();
                exit(EXIT_FAILURE);
            } else if (args[ai] == "-input") {
                input = std::string(args.at(ai + 1));
            } else if (args[ai] == "-output") {
                output = std::string(args.at(ai + 1));
            } else if (args[ai] == "-max-step-size") {
                max_step_size = std::stof(args.at(ai + 1));
            } else if (args[ai] == "-start-lr") {
                start_lr = std::stof(args.at(ai + 1));
            } else if (args[ai] == "-end-lr") {
                end_lr = std::stof(args.at(ai + 1));
            } else if (args[ai] == "-burnin-lr") {
                burnin_lr = std::stof(args.at(ai + 1));
            } else if (args[ai] == "-distribution-power") {
                distribution_power = std::stof(args.at(ai + 1));
            } else if (args[ai] == "-init-std-dev") {
                init_std_dev = std::stof(args.at(ai + 1));
            } else if (args[ai] == "-dimension") {
                dimension = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-window-size") {
                window_size = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-epochs") {
                epochs = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-burnin-epochs") {
                burnin_epochs = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-min-count") {
                min_count = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-checkpoint-interval") {
                checkpoint_interval = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-number-negatives") {
                number_negatives = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-threads") {
                threads = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-seed") {
                seed = std::stoi(args.at(ai + 1));
            } else if (args[ai] == "-t") {
                t = std::stof(args.at(ai + 1));
            } else {
                std::cerr << "Unknown argument: " << args[ai] << std::endl;
                print_help();
                exit(EXIT_FAILURE);
            }
        } catch (std::out_of_range) {
            std::cerr << args[ai] << " is missing an argument" << std::endl;
            print_help();
            exit(EXIT_FAILURE);
        }
    }
    if (input.empty() || output.empty()) {
        std::cerr << "Empty input or output path." << std::endl;
        print_help();
        exit(EXIT_FAILURE);
    }
}

void Args::print_help() {
    std::cerr
            << "  -input                  training file path\n"
            << "  -output                 output file path\n"
            << "  -min-count              minimal number of word occurences [" << min_count << "]\n"
            << "  -t                      sub-sampling threshold (0=don't subsample) [" << t << "]\n"
            << "  -start-lr               start learning rate [" << start_lr << "]\n"
            << "  -end-lr                 end learning rate [" << end_lr << "]\n"
            << "  -burnin-lr              fixed learning rate for the burnin epochs [" << burnin_lr << "]\n"
            << "  -max-step-size          max. dist to travel in one update [" << max_step_size << "]\n"
            << "  -dimension              dimension of the Minkowski ambient [" << dimension << "]\n"
            << "  -window-size            size of the context window [" << window_size << "]\n"
            << "  -init-std-dev           stddev of the hyperbolic distance from the base point for initialization [" << init_std_dev << "]\n"
            << "  -burnin-epochs          number of extra prelim epochs with burn-in learning rate [" << burnin_epochs << "]\n"
            << "  -epochs                 number of epochs with learning rate linearly decreasing from -start-lr to -end-lr [" << epochs << "]\n"
            << "  -number-negatives       number of negatives sampled [" << number_negatives << "]\n"
            << "  -distribution-power     power used to modified distribution for negative sampling [" << distribution_power << "]\n"
            << "  -checkpoint-interval    save vectors every this many epochs [" << checkpoint_interval << "]\n"
            << "  -threads                number of threads [" << threads << "]\n"
            << "  -seed                   seed for the random number generator [" << seed << "]\n"
            << "                          n.b. only deterministic if single threaded!\n";
}
}
