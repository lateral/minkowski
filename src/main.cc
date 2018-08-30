#include <iostream>

#include "minkowski.h"
#include "args.h"

using namespace minkowski;

int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
    std::shared_ptr<Args> a = std::make_shared<Args>();
    a->parse_args(args);
    Minkowski minkowski(a);
    minkowski.train();
    minkowski.save_vectors(a->output);
    return 0;
}
