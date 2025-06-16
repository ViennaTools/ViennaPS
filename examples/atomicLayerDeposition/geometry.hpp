#pragma once

#include <geometries/psMakeTrench.hpp>
#include <lsBooleanOperation.hpp>
#include <lsMakeGeometry.hpp>
#include <psDomain.hpp>
#include <psUtil.hpp>

using namespace viennacore;

template <class NumericType, int D>
void makeHAR(SmartPointer<viennaps::Domain<NumericType, D>> domain,
             NumericType gridDelta, NumericType openingDepth,
             NumericType openingWidth, NumericType gapLength,
             NumericType gapHeight, NumericType xPad,
             viennaps::Material material) {
  //   static_assert(D == 2, "This function only works in 2D");
  domain->clear();

  double bounds[2 * D];
  bounds[0] = -xPad;
  bounds[1] = openingWidth + xPad + gapLength;

  bounds[2] = -gridDelta;
  bounds[3] = openingDepth + gapHeight + gridDelta;

  typename viennals::Domain<NumericType, D>::BoundaryType boundaryCons[D];

  for (int i = 0; i < D - 1; i++) {
    boundaryCons[i] =
        viennals::Domain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
  }
  boundaryCons[D - 1] =
      viennals::Domain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

  {
    auto substrate =
        viennals::Domain<NumericType, D>::New(bounds, boundaryCons, gridDelta);
    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};
    normal[D - 1] = 1.;
    origin[D - 1] = openingDepth + gapHeight;
    viennals::MakeGeometry<NumericType, D>(
        substrate, viennals::Plane<NumericType, D>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(substrate, material);
  }

  {
    auto vertBox = viennals::Domain<NumericType, D>::New(domain->getGrid());
    NumericType minPoint[D] = {0., 0.};
    NumericType maxPoint[D] = {openingWidth,
                               gapHeight + openingDepth + gridDelta};
    viennals::MakeGeometry<NumericType, D>(
        vertBox, viennals::Box<NumericType, D>::New(minPoint, maxPoint))
        .apply();

    domain->applyBooleanOperation(
        vertBox, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
  }

  {
    auto horiBox = viennals::Domain<NumericType, D>::New(domain->getGrid());
    NumericType minPoint[D] = {openingWidth - gridDelta, 0.};
    NumericType maxPoint[D] = {openingWidth + gapLength, gapHeight};
    viennals::MakeGeometry<NumericType, D>(
        horiBox, viennals::Box<NumericType, D>::New(minPoint, maxPoint))
        .apply();

    domain->applyBooleanOperation(
        horiBox, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
  }
}

template <class NumericType, int D>
void makeT(SmartPointer<viennaps::Domain<NumericType, D>> domain,
           NumericType gridDelta, NumericType openingDepth,
           NumericType openingWidth, NumericType gapLength,
           NumericType gapHeight, NumericType xPad, viennaps::Material material,
           NumericType gapWidth = 0.) {
  //   static_assert(D == 2, "This function only works in 2D");
  domain->clear();
  typename viennals::Domain<NumericType, D>::BoundaryType boundaryCons[D];
  for (int i = 0; i < D - 1; i++) {
    boundaryCons[i] =
        viennals::Domain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
  }
  boundaryCons[D - 1] =
      viennals::Domain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

  if constexpr (D == 2) {
    double bounds[2 * D];
    bounds[0] = 0.;
    bounds[1] = openingWidth / 2. + xPad + gapLength;

    bounds[2] = -gridDelta;
    bounds[3] = openingDepth + gapHeight + gridDelta;

    {
      auto substrate = viennals::Domain<NumericType, D>::New(
          bounds, boundaryCons, gridDelta);
      NumericType normal[D] = {0.};
      NumericType origin[D] = {0.};
      normal[D - 1] = 1.;
      origin[D - 1] = openingDepth + gapHeight;
      viennals::MakeGeometry<NumericType, D>(
          substrate, viennals::Plane<NumericType, D>::New(origin, normal))
          .apply();
      domain->insertNextLevelSetAsMaterial(substrate, material);
    }

    {
      auto vertBox = viennals::Domain<NumericType, D>::New(domain->getGrid());
      NumericType minPoint[D] = {-gridDelta, 0.};
      NumericType maxPoint[D] = {openingWidth / 2.,
                                 gapHeight + openingDepth + gridDelta};
      viennals::MakeGeometry<NumericType, D>(
          vertBox, viennals::Box<NumericType, D>::New(minPoint, maxPoint))
          .apply();

      domain->applyBooleanOperation(
          vertBox, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
    }

    {
      auto horiBox = viennals::Domain<NumericType, D>::New(domain->getGrid());
      NumericType minPoint[D] = {openingWidth / 2. - gridDelta, 0.};
      NumericType maxPoint[D] = {openingWidth / 2. + gapLength, gapHeight};
      viennals::MakeGeometry<NumericType, D>(
          horiBox, viennals::Box<NumericType, D>::New(minPoint, maxPoint))
          .apply();

      domain->applyBooleanOperation(
          horiBox, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
    }
  } else {
    double bounds[2 * D];
    bounds[0] = 0.;
    bounds[1] = openingWidth / 2. + xPad + gapLength;

    bounds[2] = -gapWidth / 2. - xPad;
    bounds[3] = gapWidth / 2. + xPad;

    bounds[4] = -gridDelta;
    bounds[5] = openingDepth + gapHeight + gridDelta;

    {
      auto substrate = viennals::Domain<NumericType, D>::New(
          bounds, boundaryCons, gridDelta);
      NumericType normal[D] = {0.};
      NumericType origin[D] = {0.};
      normal[D - 1] = 1.;
      origin[D - 1] = openingDepth + gapHeight;
      viennals::MakeGeometry<NumericType, D>(
          substrate, viennals::Plane<NumericType, D>::New(origin, normal))
          .apply();
      domain->insertNextLevelSetAsMaterial(substrate, material);
    }

    {
      auto vertBox = viennals::Domain<NumericType, D>::New(domain->getGrid());
      NumericType minPoint[D] = {-gridDelta, -gapWidth / 2., 0.};
      NumericType maxPoint[D] = {openingWidth / 2., gapWidth / 2.,
                                 gapHeight + openingDepth + gridDelta};
      viennals::MakeGeometry<NumericType, D>(
          vertBox, viennals::Box<NumericType, D>::New(minPoint, maxPoint))
          .apply();

      domain->applyBooleanOperation(
          vertBox, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
    }

    {
      auto horiBox = viennals::Domain<NumericType, D>::New(domain->getGrid());
      NumericType minPoint[D] = {openingWidth / 2. - gridDelta, -gapWidth / 2.,
                                 0.};
      NumericType maxPoint[D] = {openingWidth / 2. + gapLength, gapWidth / 2.,
                                 gapHeight};
      viennals::MakeGeometry<NumericType, D>(
          horiBox, viennals::Box<NumericType, D>::New(minPoint, maxPoint))
          .apply();

      domain->applyBooleanOperation(
          horiBox, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
    }
  }
}