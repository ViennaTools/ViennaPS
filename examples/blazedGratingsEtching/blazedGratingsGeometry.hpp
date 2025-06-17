#pragma once

#include <geometries/psMakePlane.hpp>
#include <psDomain.hpp>

#include <lsBooleanOperation.hpp>
#include <lsFromSurfaceMesh.hpp>

namespace viennaps {

template <class NumericType, int D>
auto GenerateMask(const NumericType bumpWidth, const NumericType bumpHeight,
                  const int numBumps, const NumericType bumpDuty,
                  const NumericType gridDelta) {
  static_assert(D == 2, "Only 2D geometry supported for now.");
  const NumericType bumpSpacing = bumpWidth * (1. - bumpDuty) / bumpDuty;
  const NumericType xExtent = numBumps * bumpWidth / bumpDuty;
  auto domain = SmartPointer<Domain<NumericType, D>>::New(
      gridDelta, xExtent, BoundaryType::PERIODIC_BOUNDARY);

  MakePlane<NumericType, D>(domain, 0., Material::SiO2).apply();

  auto mask = viennals::Domain<NumericType, D>::New(domain->getGrid());

  // parabolic masks
  const double offset = -xExtent / 2. + bumpSpacing + bumpWidth / 2.;

  auto mesh = viennals::Mesh<NumericType>::New();

  constexpr int numNodes = 100;
  for (unsigned i = 0; i < numNodes; i++) {
    double x = -bumpWidth / 2. +
               i * bumpWidth / static_cast<NumericType>(numNodes - 1);
    double y = -4. * bumpHeight * x * x / (bumpWidth * bumpWidth) + bumpHeight;
    mesh->insertNextNode(Vec3D<NumericType>{x + offset, y, 0.});
    if (i > 0)
      mesh->insertNextLine({i - 1, i});
  }
  mesh->insertNextLine({numNodes - 1, 0});

  for (int i = 0; i < numBumps; i++) {
    auto tip = viennals::Domain<NumericType, D>::New(domain->getGrid());
    viennals::FromSurfaceMesh<NumericType, D>(tip, mesh).apply();
    viennals::TransformMesh<NumericType>(
        mesh, viennals::TransformEnum::TRANSLATION,
        Vec3D<NumericType>{bumpWidth + bumpSpacing, 0., 0.})
        .apply();
    viennals::BooleanOperation<NumericType, D>(
        mask, tip, viennals::BooleanOperationEnum::UNION)
        .apply();
  }

  domain->insertNextLevelSetAsMaterial(mask, viennaps::Material::Mask);

  return domain;
}

} // namespace viennaps
