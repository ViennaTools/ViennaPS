#pragma once

#include <cassert>
#include <lsTestAsserts.hpp>

#define PSTEST_ASSERT(condition) assert(condition)

#define PSRUN_ALL_TESTS                                                        \
  psRunTest<double, 2>();                                                      \
  psRunTest<double, 3>();                                                      \
  psRunTest<float, 2>();                                                       \
  psRunTest<float, 3>();
