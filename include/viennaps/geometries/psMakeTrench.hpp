#pragma once

#include "../psDomain.hpp"

#include <lsBooleanOperation.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsMakeGeometry.hpp>

namespace viennaps {

using namespace viennacore;

/// Generates new a trench geometry extending in the z (3D) or y (2D) direction,
/// centrally positioned at the origin with the total extent specified in the x
/// and y directions. The trench configuration may include periodic boundaries
/// in both the x and y directions. Users have the flexibility to define the
/// trench's width, depth, and incorporate tapering with a designated angle.
/// Moreover, the trench can serve as a mask, applying the specified material_
/// exclusively to the bottom while the remaining portion adopts the mask
/// material_.
template <class NumericType, int D> class MakeTrench {
  using lsDomainType = SmartPointer<viennals::Domain<NumericType, D>>;
  using psDomainType = SmartPointer<Domain<NumericType, D>>;
  using BoundaryEnum = typename viennals::Domain<NumericType, D>::BoundaryType;

  psDomainType pDomain_ = nullptr;

  const NumericType gridDelta_;
  const NumericType xExtent_;
  const NumericType yExtent_;

  const NumericType trenchWidth_;
  const NumericType trenchDepth_;
  const NumericType taperAngle_; // taper angle in degrees
  const NumericType baseHeight_;

  const bool periodicBoundary_;
  const bool halfTrench2D_;
  const bool makeMask_;
  Material material_;

public:
  MakeTrench(psDomainType domain, NumericType gridDelta, NumericType xExtent,
             NumericType yExtent, NumericType trenchWidth,
             NumericType trenchDepth, NumericType taperAngle = 0.,
             NumericType baseHeight = 0., bool periodicBoundary = false,
             bool halfTrench2D = false,
             bool makeMask = false, Material material = Material::None)
      : pDomain_(domain), gridDelta_(gridDelta), xExtent_(xExtent),
        yExtent_(yExtent), trenchWidth_(trenchWidth), trenchDepth_(trenchDepth),
        taperAngle_(taperAngle), baseHeight_(baseHeight),
        periodicBoundary_(periodicBoundary), halfTrench2D_(halfTrench2D),
        makeMask_(makeMask), material_(material) {}

  void apply() {
    pDomain_->clear();

    if (halfTrench2D_ && D == 3) {
      Logger::getInstance()
          .addWarning("Half trench is only supported in 2D.")
          .print();
      return;
    }
    if (halfTrench2D_ && periodicBoundary_) {
      Logger::getInstance()
          .addWarning("Half trench is not supported with periodic boundaries.")
          .print();
      return;
    }

    double bounds[2 * D];
    bounds[0] = -xExtent_ / 2.;
    if (halfTrench2D_) { 
      bounds[1] = 0; 
    } else { 
      bounds[1] = xExtent_ / 2.; 
    }

    if constexpr (D == 3) {
      bounds[2] = -yExtent_ / 2.;
      bounds[3] = yExtent_ / 2.;
      bounds[4] = -gridDelta_;
      bounds[5] = trenchDepth_ + gridDelta_;
    } else {
      bounds[2] = -gridDelta_;
      bounds[3] = trenchDepth_ + gridDelta_;
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

    auto substrate = lsDomainType::New(bounds, boundaryCons, gridDelta_);
    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};
    normal[D - 1] = 1.;
    origin[D - 1] = baseHeight_;
    viennals::MakeGeometry<NumericType, D>(
        substrate,
        SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    auto mask = lsDomainType::New(bounds, boundaryCons, gridDelta_);
    origin[D - 1] = trenchDepth_ + baseHeight_;
    viennals::MakeGeometry<NumericType, D>(
        mask,
        SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    auto maskAdd = lsDomainType::New(bounds, boundaryCons, gridDelta_);
    origin[D - 1] = baseHeight_;
    normal[D - 1] = -1.;
    viennals::MakeGeometry<NumericType, D>(
        maskAdd,
        SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    viennals::BooleanOperation<NumericType, D>(
        mask, maskAdd, viennals::BooleanOperationEnum::INTERSECT)
        .apply();

    auto cutout = lsDomainType::New(bounds, boundaryCons, gridDelta_);

    if (taperAngle_) {
      auto mesh = SmartPointer<viennals::Mesh<NumericType>>::New();
      const NumericType offset =
          std::tan(taperAngle_ * M_PI / 180.) * trenchDepth_;
      if constexpr (D == 2) {
        for (int i = 0; i < 4; i++) {
          std::array<NumericType, 3> node = {0., 0., 0.};
          mesh->insertNextNode(node);
        }
        mesh->nodes[0][0] = -trenchWidth_ / 2.;
        if (halfTrench2D_) { 
          mesh->nodes[1][0] = 0; 
          mesh->nodes[2][0] = 0 + offset;
        } else { 
          mesh->nodes[1][0] = trenchWidth_ / 2.; 
          mesh->nodes[2][0] = trenchWidth_ / 2. + offset;
        }
        mesh->nodes[3][0] = -trenchWidth_ / 2. - offset;

        mesh->nodes[0][1] = baseHeight_;
        mesh->nodes[1][1] = baseHeight_;
        mesh->nodes[2][1] = trenchDepth_ + baseHeight_;
        mesh->nodes[3][1] = trenchDepth_ + baseHeight_;

        mesh->insertNextLine(std::array<unsigned, 2>{0, 3});
        mesh->insertNextLine(std::array<unsigned, 2>{3, 2});
        mesh->insertNextLine(std::array<unsigned, 2>{2, 1});
        mesh->insertNextLine(std::array<unsigned, 2>{1, 0});
        viennals::FromSurfaceMesh<NumericType, D>(cutout, mesh).apply();
      } else {
        for (int i = 0; i < 8; i++) {
          std::array<NumericType, 3> node = {0., 0., 0.};
          mesh->insertNextNode(node);
        }
        mesh->nodes[0][0] = -trenchWidth_ / 2.;
        mesh->nodes[0][1] = -yExtent_ / 2. - gridDelta_;
        mesh->nodes[0][2] = baseHeight_;

        mesh->nodes[1][0] = trenchWidth_ / 2.;
        mesh->nodes[1][1] = -yExtent_ / 2. - gridDelta_;
        mesh->nodes[1][2] = baseHeight_;

        mesh->nodes[2][0] = trenchWidth_ / 2.;
        mesh->nodes[2][1] = yExtent_ / 2. + gridDelta_;
        mesh->nodes[2][2] = baseHeight_;

        mesh->nodes[3][0] = -trenchWidth_ / 2.;
        mesh->nodes[3][1] = yExtent_ / 2. + gridDelta_;
        mesh->nodes[3][2] = baseHeight_;

        mesh->nodes[4][0] = -trenchWidth_ / 2. - offset;
        mesh->nodes[4][1] = -yExtent_ / 2. - gridDelta_;
        mesh->nodes[4][2] = trenchDepth_ + baseHeight_;

        mesh->nodes[5][0] = trenchWidth_ / 2. + offset;
        mesh->nodes[5][1] = -yExtent_ / 2. - gridDelta_;
        mesh->nodes[5][2] = trenchDepth_ + baseHeight_;

        mesh->nodes[6][0] = trenchWidth_ / 2. + offset;
        mesh->nodes[6][1] = yExtent_ / 2. + gridDelta_;
        mesh->nodes[6][2] = trenchDepth_ + baseHeight_;

        mesh->nodes[7][0] = -trenchWidth_ / 2. - offset;
        mesh->nodes[7][1] = yExtent_ / 2. + gridDelta_;
        mesh->nodes[7][2] = trenchDepth_ + baseHeight_;

        mesh->insertNextTriangle(std::array<unsigned, 3>{0, 3, 1});
        mesh->insertNextTriangle(std::array<unsigned, 3>{1, 3, 2});

        mesh->insertNextTriangle(std::array<unsigned, 3>{5, 6, 4});
        mesh->insertNextTriangle(std::array<unsigned, 3>{6, 7, 4});

        mesh->insertNextTriangle(std::array<unsigned, 3>{0, 1, 5});
        mesh->insertNextTriangle(std::array<unsigned, 3>{0, 5, 4});

        mesh->insertNextTriangle(std::array<unsigned, 3>{2, 3, 6});
        mesh->insertNextTriangle(std::array<unsigned, 3>{6, 3, 7});

        mesh->insertNextTriangle(std::array<unsigned, 3>{0, 7, 3});
        mesh->insertNextTriangle(std::array<unsigned, 3>{0, 4, 7});

        mesh->insertNextTriangle(std::array<unsigned, 3>{1, 2, 6});
        mesh->insertNextTriangle(std::array<unsigned, 3>{1, 6, 5});

        viennals::FromSurfaceMesh<NumericType, D>(cutout, mesh).apply();
      }
    } else {
      NumericType minPoint[D];
      NumericType maxPoint[D];

      minPoint[0] = -trenchWidth_ / 2;
      if (halfTrench2D_) { 
        maxPoint[0] = 0; 
      } else { 
        maxPoint[0] = trenchWidth_ / 2; 
      }

      if constexpr (D == 3) {
        minPoint[1] = -yExtent_ / 2. - gridDelta_ / 2.;
        maxPoint[1] = yExtent_ / 2. + gridDelta_ / 2.;
        minPoint[2] = baseHeight_;
        maxPoint[2] = trenchDepth_ + baseHeight_;
      } else {
        minPoint[1] = baseHeight_;
        maxPoint[1] = trenchDepth_ + baseHeight_;
      }
      viennals::MakeGeometry<NumericType, D> geo(
          cutout,
          SmartPointer<viennals::Box<NumericType, D>>::New(minPoint, maxPoint));
      geo.setIgnoreBoundaryConditions(true);
      geo.apply();
    }

    viennals::BooleanOperation<NumericType, D>(
        mask, cutout, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
        .apply();

    viennals::BooleanOperation<NumericType, D>(
        substrate, mask, viennals::BooleanOperationEnum::UNION)
        .apply();

    if (material_ == Material::None) {
      if (makeMask_)
        pDomain_->insertNextLevelSet(mask);
      pDomain_->insertNextLevelSet(substrate, false);
    } else {
      if (makeMask_)
        pDomain_->insertNextLevelSetAsMaterial(mask, Material::Mask);
      pDomain_->insertNextLevelSetAsMaterial(substrate, material_, false);
    }
  }
};

} // namespace viennaps
