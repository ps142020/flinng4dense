#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <cstring>
#include <vector>
#include "FlinngIndex.h"

namespace flinng {
struct FileIO {
  FILE *fp;
  std::string fname;
};

#define WRITE_VERIFY(start, type, size, stream)              \
  {                                                          \
    size_t ret = fwrite(start, type, size, stream.fp);       \
    if (ret != (size)) {                                     \
      std::cout << "Error during write of" << stream.fname   \
      << " ret!=size " << ret << "!=" << size << " error: "  \
      << strerror(errno) << "\n";                            \
    }                                                        \
  } 

#define READ_VERIFY(start, type, size, stream)              \
  {                                                         \
    size_t ret = fread(start, type, size, stream.fp);       \
    if (ret != (size)) {                                    \
      std::cout << "Error during read of" << stream.fname   \
      << " ret!=size " << ret << "!=" << size << " error: " \
      << strerror(errno) << "\n";                           \
    }                                                       \
  } 
} //end namespace flinng