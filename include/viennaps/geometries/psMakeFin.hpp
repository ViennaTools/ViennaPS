#pragma once

#include "../psDomain.hpp"
#include "psGeometryBase.hpp"

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
template <class NumericType, int D>
class MakeFin : public GeometryBase<NumericType, D> {
  using typename GeometryBase<NumericType, D>::lsDomainType;
  using typename GeometryBase<NumericType, D>::psDomainType;
  using GeometryBase<NumericType, D>::domain_;

  const NumericType finWidth_;
  const NumericType finHeight_;
  const NumericType finTaperAngle_; // taper angle in degrees

  const NumericType maskHeight_;
  const NumericType maskTaperAngle_;

  const NumericType base_;
  const Material material_;
  const Material maskMaterial_ = Material::Mask;

public:
  MakeFin(psDomainType domain, NumericType finWidth, NumericType finHeight,
          NumericType finTaperAngle = 0., NumericType maskHeight = 0.,
          NumericType maskTaperAngle = 0., bool halfFin = false,
          Material material = Material::Si,
          Material maskMaterial = Material::Mask)
      : GeometryBase<NumericType, D>(domain), finWidth_(finWidth),
        finHeight_(finHeight), finTaperAngle_(finTaperAngle),
        maskHeight_(maskHeight), maskTaperAngle_(maskTaperAngle), base_(0.0),
        material_(material), maskMaterial_(maskMaterial) {
    if (halfFin) {
      if (domain_->getSetup().hasPeriodicBoundary()) {
        Logger::getInstance()
            .addWarning("MakeFin: Half fin cannot be created with "
                        "periodic boundaries! Creating full fin.")
            .print();
      } else {
        domain_->getSetup().bounds_[0] = 0.0;
      }
    }
  }

  MakeFin(psDomainType domain, NumericType gridDelta, NumericType xExtent,
          NumericType yExtent, NumericType finWidth, NumericType finHeight,
          NumericType taperAngle = 0., NumericType baseHeight = 0.,
          bool periodicBoundary = false, bool makeMask = false,
          Material material = Material::Si)
      : GeometryBase<NumericType, D>(domain), finWidth_(finWidth),
        finHeight_(makeMask ? 0 : finHeight),
        finTaperAngle_(makeMask ? 0 : taperAngle),
        maskHeight_(makeMask ? finHeight : 0),
        maskTaperAngle_(makeMask ? taperAngle : 0), base_(baseHeight),
        material_(material) {
    domain_->setup(gridDelta, xExtent, yExtent, periodicBoundary);
  }

  void apply() {
    domain_->clear();

    auto setup = domain_->getSetup();
    if (!setup.isValid()) {
      Logger::getInstance()
          .addWarning("MakeTrench: Domain setup is not correctly initialized.")
          .print();
      domain_->getSetup().print();
      return;
    }
    auto bounds = setup.bounds_;
    auto boundaryCons = setup.boundaryCons_;
    auto gridDelta = setup.gridDelta_;

    auto substrate = this->makeSubstrate(base_);

    if (maskHeight_ > 0.) {
      auto mask = lsDomainType::New(bounds, boundaryCons, gridDelta);
      NumericType width = finWidth_;
      if (finTaperAngle_ > 0. && finHeight_ > 0.) {
        width -= 2 * finHeight_ * std::tan(finTaperAngle_ * M_PI / 180.);
      }
      if (width > 0.) {
        if (maskTaperAngle_ > 0.) {
          getTaperedFin(mask, width, base_ + finHeight_ - gridDelta / 2.,
                        maskHeight_ + gridDelta / 2., maskTaperAngle_);
        } else {
          getFin(mask, width, base_ + finHeight_ - gridDelta / 2.,
                 maskHeight_ + gridDelta / 2.);
        }

        domain_->insertNextLevelSetAsMaterial(mask, Material::Mask);
      } else {
        Logger::getInstance()
            .addWarning("MakeFin: Mask could not be created due to invalid "
                        "fin dimensions.")
            .print();
      }
    }

    if (finHeight_ > 0.) {
      auto fin = lsDomainType::New(bounds, boundaryCons, gridDelta);
      if (finTaperAngle_ > 0.) {
        getTaperedFin(fin, finWidth_, base_ - gridDelta, finHeight_ + gridDelta,
                      finTaperAngle_);
      } else {
        getFin(fin, finWidth_, base_ - gridDelta, finHeight_ + gridDelta);
      }
      viennals::BooleanOperation<NumericType, D>(
          substrate, fin, viennals::BooleanOperationEnum::UNION)
          .apply();
    }

    domain_->insertNextLevelSetAsMaterial(substrate, material_);
  }

private:
  void getFin(lsDomainType fin, NumericType width, NumericType base,
              NumericType height) {
    NumericType minPoint[D];
    NumericType maxPoint[D];
    auto gridDelta = domain_->getGridDelta();
    auto yExtent = domain_->getSetup().yExtent();

    minPoint[0] = -width / 2;
    maxPoint[0] = width / 2;
    if constexpr (D == 3) {
      minPoint[1] = -yExtent / 2 - gridDelta;
      maxPoint[1] = yExtent / 2 + gridDelta;
    }
    minPoint[D - 1] = base;
    maxPoint[D - 1] = base + height;

    viennals::MakeGeometry<NumericType, D> geo(
        fin,
        SmartPointer<viennals::Box<NumericType, D>>::New(minPoint, maxPoint));
    geo.setIgnoreBoundaryConditions(true);
    geo.apply();
  }

  void getTaperedFin(lsDomainType fin, NumericType width, NumericType base,
                     NumericType height, NumericType angle) {
    if (angle >= 90 || angle <= -90) {
      Logger::getInstance()
          .addError("MakeFin: Taper angle must be between -90 and 90 "
                    "degrees!")
          .print();
      return;
    }
    auto gridDelta = domain_->getGridDelta();
    auto yExtent = domain_->getSetup().yExtent();

    auto boxMesh = SmartPointer<viennals::Mesh<NumericType>>::New();

    if constexpr (D == 2) {
      boxMesh->insertNextNode({-width / 2, base, 0.});
      boxMesh->insertNextNode({width / 2, base, 0.});
      boxMesh->insertNextLine({1, 0});

      NumericType offSet = height * std::tan(angle * M_PI / 180.);
      if (offSet >= width / 2) {
        NumericType top = base + width / 2 / std::tan(angle * M_PI / 180.);
        boxMesh->insertNextNode({0., top, 0.});
        boxMesh->insertNextLine({2, 1});
        boxMesh->insertNextLine({0, 2});
      } else {
        boxMesh->insertNextNode({width / 2 - offSet, base + height, 0.});
        boxMesh->insertNextNode({-width / 2 + offSet, base + height, 0.});
        boxMesh->insertNextLine({2, 1});
        boxMesh->insertNextLine({3, 2});
        boxMesh->insertNextLine({0, 3});
      }
    } else { // 3D
      boxMesh->insertNextNode({-width / 2, yExtent / 2 + gridDelta, base});
      boxMesh->insertNextNode({width / 2, yExtent / 2 + gridDelta, base});

      NumericType offSet = height * std::tan(angle * M_PI / 180.);

      if (offSet >= width / 2) { // single top node
        NumericType top = base + width / 2 / std::tan(angle * M_PI / 180.);
        boxMesh->insertNextNode({0., yExtent / 2 + gridDelta, top});

        // shifted nodes by y extent
        boxMesh->insertNextNode({-width / 2, -yExtent / 2 - gridDelta, base});
        boxMesh->insertNextNode({width / 2, -yExtent / 2 - gridDelta, base});
        boxMesh->insertNextNode({0., -yExtent / 2 - gridDelta, top});

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
        boxMesh->insertNextNode(
            {width / 2 - offSet, yExtent / 2 + gridDelta, base + height});
        boxMesh->insertNextNode(
            {-width / 2 + offSet, yExtent / 2 + gridDelta, base + height});

        // shifted nodes by y extent
        boxMesh->insertNextNode(
            {-width / 2, -yExtent / 2 - gridDelta, base - gridDelta});
        boxMesh->insertNextNode(
            {width / 2, -yExtent / 2 - gridDelta, base - gridDelta});
        boxMesh->insertNextNode(
            {width / 2 - offSet, -yExtent / 2 - gridDelta, base + height});
        boxMesh->insertNextNode(
            {-width / 2 + offSet, -yExtent / 2 - gridDelta, base + height});

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
    }
    viennals::FromSurfaceMesh<NumericType, D>(fin, boxMesh).apply();
  }
};

} // namespace viennaps
