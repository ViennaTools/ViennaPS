#pragma once

#include <geometries/psMakePlane.hpp>
#include <psDomain.hpp>

#include <lsBooleanOperation.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsMakeGeometry.hpp>

namespace viennaps {

template <class NumericType, int D>
auto generateMask(const NumericType bumpWidth, const NumericType bumpHeight,
                  const int numBumps, const NumericType bumpDuty,
                  const NumericType gridDelta, const NumericType yExtent = 0) {

  const NumericType bumpSpacing = bumpWidth * (1. - bumpDuty) / bumpDuty;
  const NumericType xExtent = numBumps * bumpWidth / bumpDuty;
  auto domain = SmartPointer<Domain<NumericType, D>>::New(
      gridDelta, xExtent, yExtent, BoundaryType::PERIODIC_BOUNDARY);

  MakePlane<NumericType, D>(domain, 0., Material::SiO2).apply();

  if constexpr (D == 2) {
    auto mask =
        SmartPointer<viennals::Domain<NumericType, D>>::New(domain->getGrid());

    if (bumpHeight == 0.) {
      // circular masks
      double origin[D] = {-xExtent / 2. + bumpSpacing + bumpWidth / 2., 0.};
      for (int i = 0; i < numBumps; i++) {
        auto ball = SmartPointer<viennals::Domain<NumericType, D>>::New(
            domain->getGrid());
        viennals::MakeGeometry<NumericType, D>(
            ball, SmartPointer<viennals::Sphere<NumericType, D>>::New(
                      origin, bumpWidth / 2.))
            .apply();

        origin[0] += bumpWidth + bumpSpacing;

        viennals::BooleanOperation<NumericType, D>(
            mask, ball, viennals::BooleanOperationEnum::UNION)
            .apply();
      }

      viennals::BooleanOperation<NumericType, D>(
          mask, domain->getLevelSets().back(),
          viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();
    } else {
      // parabolic masks
      const double offset = -xExtent / 2. + bumpSpacing + bumpWidth / 2.;

      auto mesh = SmartPointer<viennals::Mesh<NumericType>>::New();

      auto maskContour = [=](double x) {
        return -4. * bumpHeight * x * x / (bumpWidth * bumpWidth) + bumpHeight;
      };

      for (unsigned i = 0; i < 100; i++) {
        double x = -bumpWidth / 2. + i * bumpWidth / 99.;
        double y = maskContour(x);
        mesh->insertNextNode({x + offset, y, 0.});
        if (i > 0)
          mesh->insertNextLine({i - 1, i});
      }
      mesh->insertNextLine({99, 0});

      for (int i = 0; i < numBumps; i++) {

        auto tip = SmartPointer<viennals::Domain<NumericType, D>>::New(
            domain->getGrid());
        viennals::FromSurfaceMesh<NumericType, D>(tip, mesh).apply();

        for (int j = 0; j < mesh->getNodes().size(); j++) {
          mesh->nodes[j][0] += bumpWidth + bumpSpacing;
        }

        viennals::BooleanOperation<NumericType, D>(
            mask, tip, viennals::BooleanOperationEnum::UNION)
            .apply();
      }
    }

    domain->insertNextLevelSetAsMaterial(mask, viennaps::Material::Mask);
  } else {
    Logger::getInstance()
        .addWarning("Only 2D geometry supported for now.")
        .print();
  }

  return domain;
}

} // namespace viennaps
