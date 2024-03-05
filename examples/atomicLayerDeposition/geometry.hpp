#pragma once

#include <lsBooleanOperation.hpp>
#include <lsMakeGeometry.hpp>
#include <psDomain.hpp>
#include <psMakeTrench.hpp>
#include <psUtils.hpp>

template <class NumericType, int D>
void makeLShape(psSmartPointer<psDomain<NumericType, D>> domain,
                psUtils::Parameters &params, psMaterial material) {
  static_assert(D == 2, "This function only works in 2D");
  domain->clear();
  const auto gridDelta = params.get("gridDelta");

  double bounds[2 * D];
  bounds[0] = -params.get("verticalWidth") / 2. - params.get("xPad");
  bounds[1] = params.get("verticalWidth") / 2. + params.get("xPad") +
              params.get("horizontalWidth");

  bounds[2] = -gridDelta;
  bounds[3] = params.get("verticalDepth") + gridDelta;

  typename lsDomain<NumericType, D>::BoundaryType boundaryCons[D];

  for (int i = 0; i < D - 1; i++) {
    boundaryCons[i] =
        lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
  }
  boundaryCons[D - 1] =
      lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

  {
    auto substrate = lsSmartPointer<lsDomain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);
    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};
    normal[D - 1] = 1.;
    origin[D - 1] = params.get("verticalDepth");
    lsMakeGeometry<NumericType, D>(
        substrate, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(substrate, material);
  }

  {
    auto vertBox =
        lsSmartPointer<lsDomain<NumericType, D>>::New(domain->getGrid());
    NumericType minPoint[D] = {-params.get("verticalWidth") / 2.0, 0.};
    NumericType maxPoint[D] = {params.get("verticalWidth") / 2.0,
                               params.get("verticalDepth")};
    lsMakeGeometry<NumericType, D>(
        vertBox, lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
        .apply();

    domain->applyBooleanOperation(vertBox,
                                  lsBooleanOperationEnum::RELATIVE_COMPLEMENT);
  }

  {
    auto horiBox =
        lsSmartPointer<lsDomain<NumericType, D>>::New(domain->getGrid());
    NumericType minPoint[D] = {-params.get("verticalWidth") / 2.0, 0.};
    NumericType maxPoint[D] = {-params.get("verticalWidth") / 2.0 +
                                   params.get("horizontalWidth"),
                               params.get("horizontalHeight")};

    lsMakeGeometry<NumericType, D>(
        horiBox, lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
        .apply();

    domain->applyBooleanOperation(horiBox,
                                  lsBooleanOperationEnum::RELATIVE_COMPLEMENT);
  }
}