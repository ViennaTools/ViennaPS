#pragma once

#include <lsBooleanOperation.hpp>
#include <lsMakeGeometry.hpp>

#include <geometries/psMakeTrench.hpp>
#include <psDomain.hpp>
#include <psUtils.hpp>

template <class NumericType, int D>
void makeLShape(psSmartPointer<psDomain<NumericType, D>> domain,
                psUtils::Parameters &params, psMaterial material) {
  static_assert(D == 2, "This function only works in 2D");
  domain->clear();
  const auto gridDelta = params.get("gridDelta");

  double bounds[2 * D];
  bounds[0] = -params.get("verticalWidth") / 2. - params.get("xPad");
  bounds[1] = -params.get("verticalWidth") / 2. + params.get("xPad") +
              params.get("horizontalWidth");
  bounds[2] = -gridDelta;
  bounds[3] = params.get("verticalDepth") + gridDelta;

  typename ls::Domain<NumericType, D>::BoundaryType boundaryCons[D];

  for (int i = 0; i < D - 1; i++) {
    boundaryCons[i] =
        ls::Domain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
  }
  boundaryCons[D - 1] =
      ls::Domain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

  {
    auto substrate = ps::SmartPointer<ls::Domain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);
    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};
    normal[D - 1] = 1.;
    origin[D - 1] = params.get("verticalDepth");
    ls::MakeGeometry<NumericType, D>(
        substrate,
        ps::SmartPointer<ls::Plane<NumericType, D>>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(substrate, material);
  }

  {
    auto vertBox =
        ps::SmartPointer<ls::Domain<NumericType, D>>::New(domain->getGrid());
    NumericType minPoint[D] = {-params.get("verticalWidth") / 2.0, 0.};
    NumericType maxPoint[D] = {params.get("verticalWidth") / 2.0,
                               params.get("verticalDepth")};
    ls::MakeGeometry<NumericType, D>(
        vertBox,
        ps::SmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
        .apply();

    domain->applyBooleanOperation(vertBox,
                                  lsBooleanOperationEnum::RELATIVE_COMPLEMENT);
  }

  {
    auto horiBox =
        ps::SmartPointer<ls::Domain<NumericType, D>>::New(domain->getGrid());
    NumericType minPoint[D] = {-params.get("verticalWidth") / 2.0, 0.};
    NumericType maxPoint[D] = {-params.get("verticalWidth") / 2.0 +
                                   params.get("horizontalWidth"),
                               params.get("horizontalHeight")};

    ls::MakeGeometry<NumericType, D>(
        horiBox,
        ps::SmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
        .apply();

    domain->applyBooleanOperation(horiBox,
                                  lsBooleanOperationEnum::RELATIVE_COMPLEMENT);
  }
}
