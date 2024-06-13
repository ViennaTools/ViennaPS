#pragma once

#include "../psDomain.hpp"

#include <lsMakeGeometry.hpp>

#include <vcLogger.hpp>

namespace viennaps {

using namespace viennacore;

/// This class provides a simple way to create a plane in a level set. It can be
/// used to create a substrate of any material_. The plane can be added to an
/// already existing geometry or a new geometry can be created. The plane is
/// created with normal direction in the positive z direction in 3D and positive
/// y direction in 2D. The plane is centered around the origin with the total
/// specified extent and height. The plane can have a periodic boundary in the x
/// and y (only 3D) direction.
template <class NumericType, int D> class MakePlane {
  using LSPtrType = SmartPointer<viennals::Domain<NumericType, D>>;
  using psDomainType = SmartPointer<Domain<NumericType, D>>;
  using BoundaryEnum = typename viennals::Domain<NumericType, D>::BoundaryType;

  psDomainType pDomain_ = nullptr;

  const NumericType gridDelta_ = 0.;
  const NumericType xExtent_ = 0.;
  const NumericType yExtent_ = 0.;
  const NumericType baseHeight_;

  const bool periodicBoundary_ = false;
  const Material material_;

  const bool add_;

public:
  // Adds a plane to an already existing geometry.
  MakePlane(psDomainType domain, NumericType baseHeight = 0.,
            Material material = Material::None)
      : pDomain_(domain), baseHeight_(baseHeight), material_(material),
        add_(true) {}

  // Creates a new geometry with a plane.
  MakePlane(psDomainType domain, NumericType gridDelta, NumericType xExtent,
            NumericType yExtent, NumericType baseHeight,
            bool periodicBoundary = false, Material material = Material::None)
      : pDomain_(domain), gridDelta_(gridDelta), xExtent_(xExtent),
        yExtent_(yExtent), baseHeight_(baseHeight),
        periodicBoundary_(periodicBoundary), material_(material), add_(false) {}

  void apply() {
    if (add_) {
      if (!pDomain_->getLevelSets().back()) {
        Logger::getInstance()
            .addWarning("MakePlane: Plane can only be added to already "
                        "existing geometry.")
            .print();
        return;
      }
    } else {
      pDomain_->clear();
    }

    double bounds[2 * D];
    bounds[0] = -xExtent_ / 2.;
    bounds[1] = xExtent_ / 2.;

    if constexpr (D == 3) {
      bounds[2] = -yExtent_ / 2.;
      bounds[3] = yExtent_ / 2.;
      bounds[4] = -gridDelta_;
      bounds[5] = gridDelta_;
    } else {
      bounds[2] = -gridDelta_;
      bounds[3] = gridDelta_;
    }

    BoundaryEnum boundaryCons[D];
    for (int i = 0; i < D - 1; i++) {
      if (periodicBoundary_) {
        boundaryCons[i] = BoundaryEnum::PERIODIC_BOUNDARY;
      } else {
        boundaryCons[i] = BoundaryEnum::REFLECTIVE_BOUNDARY;
      }
    }
    boundaryCons[D - 1] = BoundaryEnum::INFINITE_BOUNDARY;

    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};
    normal[D - 1] = 1.;
    origin[D - 1] = baseHeight_;

    if (add_) {
      auto substrate = LSPtrType::New(pDomain_->getGrid());
      viennals::MakeGeometry<NumericType, D>(
          substrate,
          SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
          .apply();
      if (material_ == Material::None) {
        pDomain_->insertNextLevelSet(substrate);
      } else {
        pDomain_->insertNextLevelSetAsMaterial(substrate, material_);
      }
    } else {
      auto substrate = LSPtrType::New(bounds, boundaryCons, gridDelta_);
      viennals::MakeGeometry<NumericType, D>(
          substrate,
          SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
          .apply();
      if (material_ == Material::None) {
        pDomain_->insertNextLevelSet(substrate);
      } else {
        pDomain_->insertNextLevelSetAsMaterial(substrate, material_);
      }
    }
  }
};

} // namespace viennaps
