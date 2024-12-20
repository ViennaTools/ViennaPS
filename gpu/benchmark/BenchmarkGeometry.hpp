#pragma once

#include <geometries/psMakeHole.hpp>
#include <geometries/psMakeTrench.hpp>
#include <psDomain.hpp>

#define MAKE_GEO Trench
#define GRID_DELTA 0.1
#define DIM 3

template <class NumericType> auto Trench() {
  NumericType xExtent = 10.;
  NumericType yExtent = 5.;
  NumericType width = 5.;
  NumericType depth = 15.;

  using namespace viennaps;
  auto domain = SmartPointer<Domain<NumericType, DIM>>::New();
  MakeTrench<NumericType, DIM>(domain, GRID_DELTA, xExtent, yExtent, width,
                               depth, 0., GRID_DELTA / 2., false, true,
                               Material::Si)
      .apply();
  return domain;
}

template <class NumericType> auto Hole() {
  NumericType xExtent = 10.;
  NumericType yExtent = 10.;
  NumericType radius = 3.;
  NumericType depth = 15.;

  using namespace viennaps;
  auto domain = SmartPointer<Domain<NumericType, DIM>>::New();
  MakeHole<NumericType, DIM>(domain, GRID_DELTA, xExtent, yExtent, radius,
                             depth, 0., GRID_DELTA / 2., false, true,
                             Material::Si)
      .apply();
  return domain;
}