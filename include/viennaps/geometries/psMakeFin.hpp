#pragma once

#include "../psDomain.hpp"

#include <lsBooleanOperation.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsMakeGeometry.hpp>

#include <vcLogger.hpp>

namespace viennaps {

using namespace viennacore;

/// Generates a new fin geometry extending in the z (3D) or y (2D) direction,
/// centered at the origin with specified dimensions in the x and y directions.
/// The fin may incorporate periodic boundaries in the x and y directions. Users
/// can define the width and height of the fin, and it can function as a mask,
/// with the specified material exclusively applied to the bottom of the fin,
/// while the upper portion adopts the mask material.
template <class NumericType, int D> class MakeFin {
  using lsDomainType = SmartPointer<viennals::Domain<NumericType, D>>;
  using psDomainType = SmartPointer<Domain<NumericType, D>>;
  using BoundaryEnum = typename viennals::Domain<NumericType, D>::BoundaryType;

  psDomainType pDomain_ = nullptr;

  const NumericType gridDelta_;
  const NumericType xExtent_;
  const NumericType yExtent_;

  const NumericType finWidth_;
  const NumericType finHeight_;
  const NumericType taperAngle_; // taper angle in degrees
  const NumericType baseHeight_;

  const bool periodicBoundary_;
  const bool makeMask_;
  const Material material_;

public:
  MakeFin(psDomainType domain, NumericType gridDelta, NumericType xExtent,
          NumericType yExtent, NumericType finWidth, NumericType finHeight,
          NumericType taperAngle = 0., NumericType baseHeight = 0.,
          bool periodicBoundary = false, bool makeMask = false,
          Material material = Material::None)
      : pDomain_(domain), gridDelta_(gridDelta), xExtent_(xExtent),
        yExtent_(yExtent), finWidth_(finWidth), finHeight_(finHeight),
        taperAngle_(taperAngle), baseHeight_(baseHeight),
        periodicBoundary_(periodicBoundary), makeMask_(makeMask),
        material_(material) {}

  void apply() {
    pDomain_->clear();

    if constexpr (D == 3) {
      double bounds[2 * D] = {-xExtent_ / 2, xExtent_ / 2, -yExtent_ / 2,
                              yExtent_ / 2,  -1.,          1.};

      BoundaryEnum boundaryConds[D] = {BoundaryEnum::REFLECTIVE_BOUNDARY,
                                       BoundaryEnum::REFLECTIVE_BOUNDARY,
                                       BoundaryEnum::INFINITE_BOUNDARY};
      if (periodicBoundary_) {
        boundaryConds[0] = BoundaryEnum::PERIODIC_BOUNDARY;
        boundaryConds[1] = BoundaryEnum::PERIODIC_BOUNDARY;
      }

      auto substrate = lsDomainType::New(bounds, boundaryConds, gridDelta_);
      NumericType normal[D] = {0., 0., 1.};
      NumericType origin[D] = {0., 0., baseHeight_};
      viennals::MakeGeometry<NumericType, D>(
          substrate,
          SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
          .apply();

      auto mask = lsDomainType::New(bounds, boundaryConds, gridDelta_);

      if (taperAngle_ == 0.) {
        NumericType minPoint[D] = {-finWidth_ / 2.f,
                                   -yExtent_ / 2.f - gridDelta_,
                                   baseHeight_ - gridDelta_};
        NumericType maxPoint[D] = {finWidth_ / 2.f, yExtent_ / 2.f + gridDelta_,
                                   baseHeight_ + finHeight_};

        viennals::MakeGeometry<NumericType, D> geo(
            mask, SmartPointer<viennals::Box<NumericType, D>>::New(minPoint,
                                                                   maxPoint));
        geo.setIgnoreBoundaryConditions(true);
        geo.apply();

      } else {
        if (taperAngle_ >= 90 || taperAngle_ <= -90) {
          Logger::getInstance()
              .addError("MakeFin: Taper angle must be between -90 and 90 "
                        "degrees!")
              .print();
          return;
        }

        auto boxMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
        boxMesh->insertNextNode({-finWidth_ / 2, yExtent_ / 2 + gridDelta_,
                                 baseHeight_ - gridDelta_});
        boxMesh->insertNextNode({finWidth_ / 2, yExtent_ / 2 + gridDelta_,
                                 baseHeight_ - gridDelta_});

        NumericType taperAngleRad = taperAngle_ * M_PI / 180.;
        NumericType offSet = finHeight_ * std::tan(taperAngleRad);
        if (offSet >= finWidth_ / 2) {
          boxMesh->insertNextNode(
              {0., yExtent_ / 2 + gridDelta_,
               baseHeight_ + finWidth_ / 2 / std::tan(taperAngleRad)});

          // shifted nodes by y extent
          boxMesh->insertNextNode({-finWidth_ / 2, -yExtent_ / 2 - gridDelta_,
                                   baseHeight_ - gridDelta_});
          boxMesh->insertNextNode({finWidth_ / 2, -yExtent_ / 2 - gridDelta_,
                                   baseHeight_ - gridDelta_});
          boxMesh->insertNextNode(
              {0., -yExtent_ / 2 - gridDelta_,
               baseHeight_ + finWidth_ / 2 / std::tan(taperAngleRad)});

          // triangles
          boxMesh->insertNextTriangle({0, 2, 1}); // front
          boxMesh->insertNextTriangle({3, 4, 5}); // back
          boxMesh->insertNextTriangle({0, 1, 3}); // bottom
          boxMesh->insertNextTriangle({1, 4, 3}); // bottom
          boxMesh->insertNextTriangle({1, 2, 5}); // right
          boxMesh->insertNextTriangle({1, 5, 4}); // right
          boxMesh->insertNextTriangle({0, 3, 2}); // left
          boxMesh->insertNextTriangle({3, 5, 2}); // left
        } else {
          boxMesh->insertNextNode({finWidth_ / 2 - offSet,
                                   yExtent_ / 2 + gridDelta_,
                                   baseHeight_ + finHeight_});
          boxMesh->insertNextNode({-finWidth_ / 2 + offSet,
                                   yExtent_ / 2 + gridDelta_,
                                   baseHeight_ + finHeight_});

          // shifted nodes by y extent
          boxMesh->insertNextNode({-finWidth_ / 2, -yExtent_ / 2 - gridDelta_,
                                   baseHeight_ - gridDelta_});
          boxMesh->insertNextNode({finWidth_ / 2, -yExtent_ / 2 - gridDelta_,
                                   baseHeight_ - gridDelta_});
          boxMesh->insertNextNode({finWidth_ / 2 - offSet,
                                   -yExtent_ / 2 - gridDelta_,
                                   baseHeight_ + finHeight_});
          boxMesh->insertNextNode({-finWidth_ / 2 + offSet,
                                   -yExtent_ / 2 - gridDelta_,
                                   baseHeight_ + finHeight_});

          // triangles
          boxMesh->insertNextTriangle({0, 3, 1}); // front
          boxMesh->insertNextTriangle({1, 3, 2}); // front
          boxMesh->insertNextTriangle({4, 5, 6}); // back
          boxMesh->insertNextTriangle({4, 6, 7}); // back
          boxMesh->insertNextTriangle({0, 1, 4}); // bottom
          boxMesh->insertNextTriangle({1, 5, 4}); // bottom
          boxMesh->insertNextTriangle({1, 2, 5}); // right
          boxMesh->insertNextTriangle({2, 6, 5}); // right
          boxMesh->insertNextTriangle({0, 4, 3}); // left
          boxMesh->insertNextTriangle({3, 4, 7}); // left
          boxMesh->insertNextTriangle({3, 7, 2}); // top
          boxMesh->insertNextTriangle({2, 7, 6}); // top
        }
        viennals::FromSurfaceMesh<NumericType, D>(mask, boxMesh).apply();
      }

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
    } else if constexpr (D == 2) {

      double bounds[2 * D] = {-xExtent_ / 2, xExtent_ / 2,
                              baseHeight_ - gridDelta_,
                              baseHeight_ + finHeight_ + gridDelta_};

      BoundaryEnum boundaryConds[D] = {BoundaryEnum::REFLECTIVE_BOUNDARY,
                                       BoundaryEnum::INFINITE_BOUNDARY};
      if (periodicBoundary_) {
        boundaryConds[0] = BoundaryEnum::PERIODIC_BOUNDARY;
      }

      auto substrate = lsDomainType::New(bounds, boundaryConds, gridDelta_);
      NumericType normal[D] = {0., 1.};
      NumericType origin[D] = {0., baseHeight_};
      viennals::MakeGeometry<NumericType, D>(
          substrate,
          SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
          .apply();

      auto mask = lsDomainType::New(bounds, boundaryConds, gridDelta_);

      if (taperAngle_ == 0.) {
        NumericType minPoint[D] = {-finWidth_ / 2, baseHeight_ - gridDelta_};
        NumericType maxPoint[D] = {finWidth_ / 2, baseHeight_ + finHeight_};
        viennals::MakeGeometry<NumericType, D> geo(
            mask, SmartPointer<viennals::Box<NumericType, D>>::New(minPoint,
                                                                   maxPoint));
        geo.setIgnoreBoundaryConditions(true);
        geo.apply();
      } else {
        if (taperAngle_ >= 90 || taperAngle_ <= -90) {
          Logger::getInstance()
              .addError("MakeFin: Taper angle must be between -90 and 90 "
                        "degrees!")
              .print();
          return;
        }

        auto boxMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
        boxMesh->insertNextNode({-finWidth_ / 2, baseHeight_ - gridDelta_});
        boxMesh->insertNextNode({finWidth_ / 2, baseHeight_ - gridDelta_});
        boxMesh->insertNextLine({1, 0});

        NumericType taperAngleRad = taperAngle_ * M_PI / 180.;
        NumericType offSet = finHeight_ * std::tan(taperAngleRad);
        if (offSet >= finWidth_ / 2) {
          boxMesh->insertNextNode(
              {0., baseHeight_ + finWidth_ / 2 / std::tan(taperAngleRad)});
          boxMesh->insertNextLine({2, 1});
          boxMesh->insertNextLine({0, 2});
        } else {
          boxMesh->insertNextNode(
              {finWidth_ / 2 - offSet, baseHeight_ + finHeight_});
          boxMesh->insertNextNode(
              {-finWidth_ / 2 + offSet, baseHeight_ + finHeight_});
          boxMesh->insertNextLine({2, 1});
          boxMesh->insertNextLine({3, 2});
          boxMesh->insertNextLine({0, 3});
        }

        viennals::FromSurfaceMesh<NumericType, D>(mask, boxMesh).apply();
      }

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
  }
};

} // namespace viennaps
