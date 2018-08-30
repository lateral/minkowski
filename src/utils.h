#pragma once

#include <fstream>

namespace minkowski {

namespace utils {

  int64_t size(std::ifstream&);
  void seek(std::ifstream&, int64_t);
}

}
