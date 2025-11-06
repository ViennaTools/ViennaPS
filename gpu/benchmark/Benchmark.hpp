#pragma once

#include <geometries/psMakeHole.hpp>
#include <geometries/psMakeTrench.hpp>
#include <psDomain.hpp>

#define MAKE_GEO Hole
#define DEFAULT_GRID_DELTA 0.1
#define DEFAULT_STICKING 0.1
#define DIM 3

using TranslatorType = std::unordered_map<unsigned long, unsigned long>;

template <class NumericType>
auto Trench(NumericType gridDelta = DEFAULT_GRID_DELTA) {
  NumericType xExtent = 10.;
  NumericType yExtent = 10.;
  NumericType width = 5.;
  NumericType depth = 25.;

  using namespace viennaps;
  auto domain = Domain<NumericType, DIM>::New(
      gridDelta, xExtent, yExtent, BoundaryType::REFLECTIVE_BOUNDARY);
  MakeTrench<NumericType, DIM>(domain, width, depth).apply().apply();
  return domain;
}

template <class NumericType>
auto Hole(NumericType gridDelta = DEFAULT_GRID_DELTA) {
  NumericType xExtent = 10.;
  NumericType yExtent = 10.;
  NumericType radius = 3.0;
  NumericType depth = 30.;

  using namespace viennaps;
  auto domain = Domain<NumericType, DIM>::New(
      gridDelta, xExtent, yExtent, BoundaryType::REFLECTIVE_BOUNDARY);
  MakeHole<NumericType, DIM>(domain, radius, depth).apply();
  return domain;
}

template <class NumericType, int N>
std::array<NumericType, N> linspace(NumericType start, NumericType end) {
  std::array<NumericType, N> arr{};
  NumericType step = (end - start) / static_cast<NumericType>(N - 1);
  for (int i = 0; i < N; ++i) {
    arr[i] = start + i * step;
  }
  return arr;
}
