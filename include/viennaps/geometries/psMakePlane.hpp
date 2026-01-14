#pragma once

#include "../psDomain.hpp"
#include "psGeometryFactory.hpp"

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
  using psDomainType = SmartPointer<Domain<NumericType, D>>;

  psDomainType domain_;
  GeometryFactory<NumericType, D> geometryFactory_;

  const NumericType baseHeight_;
  const Material material_;
  const bool add_;

public:
  // Adds a plane to an already existing geometry.
  MakePlane(psDomainType domain, NumericType baseHeight = 0.,
            Material material = Material::Si, bool addToExisting = false)
      : domain_(domain), geometryFactory_(domain->getSetup(), __func__),
        baseHeight_(baseHeight), material_(material), add_(addToExisting) {}

  // Creates a new geometry with a plane.
  MakePlane(psDomainType domain, NumericType gridDelta, NumericType xExtent,
            NumericType yExtent, NumericType baseHeight,
            bool periodicBoundary = false, Material material = Material::Si)
      : domain_(domain), geometryFactory_(domain->getSetup(), __func__),
        baseHeight_(baseHeight), material_(material), add_(false) {
    domain_->setup(gridDelta, xExtent, yExtent,
                   periodicBoundary ? BoundaryType::PERIODIC_BOUNDARY
                                    : BoundaryType::REFLECTIVE_BOUNDARY);
    geometryFactory_.setup(domain_->getSetup());
  }

  void apply() {
    if (add_) {
      if (!domain_->getLevelSets().back()) {
        VIENNACORE_LOG_ERROR(
            "MakePlane: No existing geometry to add plane to.");
        return;
      }
    } else {
      domain_->clear();
    }

    domain_->getSetup().check();
    auto substrate = geometryFactory_.makeSubstrate(baseHeight_);
    domain_->insertNextLevelSetAsMaterial(substrate, material_);
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(MakePlane)

} // namespace viennaps
