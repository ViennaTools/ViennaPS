#pragma once

#include <cassert>
#include <lsTestAsserts.hpp>

#define PSTEST_ASSERT(condition) assert(condition)

#define PSRUN_3D_TESTS                                                         \
  psRunTest<double, 3>();                                                      \
  psRunTest<float, 3>();

#define PSRUN_2D_TESTS                                                         \
  psRunTest<double, 2>();                                                      \
  psRunTest<float, 2>();

#define PSRUN_ALL_TESTS                                                        \
  psRunTest<double, 2>();                                                      \
  psRunTest<double, 3>();                                                      \
  psRunTest<float, 2>();                                                       \
  psRunTest<float, 3>();