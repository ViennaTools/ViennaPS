#pragma once

#include "../geometries/psGeometryBase.hpp"
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
template <class NumericType, int D>
class MakeTrench : public GeometryBase<NumericType, D> {
  using typename GeometryBase<NumericType, D>::lsDomainType;
  using typename GeometryBase<NumericType, D>::psDomainType;
  using GeometryBase<NumericType, D>::domain_;

  const NumericType trenchWidth_;
  const NumericType trenchDepth_;
  const NumericType trenchTaperAngle_; // angle in degrees

  const NumericType maskHeight_;
  const NumericType maskTaperAngle_;

  const NumericType base_;
  const Material material_;

public:
  MakeTrench(psDomainType domain, NumericType trenchWidth,
             NumericType trenchDepth, NumericType trenchTaperAngle = 0,
             NumericType maskHeight = 0, NumericType maskTaperAngle = 0,
             bool halfTrench = false, Material material = Material::Si)
      : GeometryBase<NumericType, D>(domain), trenchWidth_(trenchWidth),
        trenchDepth_(trenchDepth), trenchTaperAngle_(trenchTaperAngle),
        maskHeight_(maskHeight), maskTaperAngle_(maskTaperAngle), base_(0.0),
        material_(material) {
    if (halfTrench) {
      if (domain_->getSetup().hasPeriodicBoundary()) {
        Logger::getInstance()
            .addWarning("MakeTrench: Half trench cannot be created with "
                        "periodic boundaries! Creating full trench.")
            .print();
      } else {
        domain->getSetup().bounds_[0] = 0.0;
      }
    }
  }

  MakeTrench(psDomainType domain, NumericType gridDelta, NumericType xExtent,
             NumericType yExtent, NumericType trenchWidth,
             NumericType trenchDepth, NumericType taperAngle = 0.,
             NumericType base = 0., bool periodicBoundary = false,
             bool makeMask = false, Material material = Material::Si)
      : GeometryBase<NumericType, D>(domain), trenchWidth_(trenchWidth),
        trenchDepth_(makeMask ? 0 : trenchDepth),
        trenchTaperAngle_(makeMask ? 0 : taperAngle),
        maskHeight_(makeMask ? trenchDepth : 0),
        maskTaperAngle_(makeMask ? taperAngle : 0), base_(base),
        material_(material) {
    domain_->setup(gridDelta, xExtent, yExtent, periodicBoundary);
  }

  void apply() {
    domain_->clear(); // this does not clear the setup
    auto setup = domain_->getSetup();
    auto bounds = setup.bounds_;
    auto boundaryCons = setup.boundaryCons_;
    auto gridDelta = setup.gridDelta_;
    if (gridDelta == 0.0) {
      Logger::getInstance()
          .addWarning("MakeTrench: Domain setup is not initialized.")
          .print();
      return;
    }

    auto substrate = this->makeSubstrate(base_);

    if (maskHeight_ > 0.) {
      auto mask = this->makeMask(base_, maskHeight_);
      auto cutout = lsDomainType::New(bounds, boundaryCons, gridDelta);
      NumericType width = trenchWidth_;
      if (trenchTaperAngle_ > 0. && trenchDepth_ > 0.) {
        width += 2 * std::tan(trenchTaperAngle_ * M_PI / 180.) *
                 (trenchDepth_ + gridDelta);
      }
      if (maskTaperAngle_ > 0.) {
        getTaperedCutout(cutout, width, base_, maskHeight_, maskTaperAngle_);
      } else {
        getCutout(cutout, width, base_, maskHeight_);
      }

      viennals::BooleanOperation<NumericType, D>(
          mask, cutout, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();

      domain_->insertNextLevelSetAsMaterial(mask, Material::Mask);
    }

    if (trenchDepth_ > 0.) {
      auto cutout = lsDomainType::New(bounds, boundaryCons, gridDelta);
      if (trenchTaperAngle_ > 0.) {
        getTaperedCutout(cutout, trenchWidth_, base_ - trenchDepth_,
                         trenchDepth_ + gridDelta, trenchTaperAngle_);
      } else {
        getCutout(cutout, trenchWidth_, base_ - trenchDepth_,
                  trenchDepth_ + gridDelta);
      }
      viennals::BooleanOperation<NumericType, D>(
          substrate, cutout,
          viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();
    }

    domain_->insertNextLevelSetAsMaterial(substrate, material_);
  }

private:
  void getCutout(SmartPointer<viennals::Domain<NumericType, D>> cutout,
                 NumericType width, NumericType base, NumericType height) {
    NumericType minPoint[D];
    NumericType maxPoint[D];
    auto gridDelta = domain_->getGridDelta();
    auto yExtent = domain_->getSetup().yExtent();

    minPoint[0] = -width / 2;
    maxPoint[0] = width / 2;
    if constexpr (D == 3) {
      minPoint[1] = -yExtent / 2. - gridDelta;
      maxPoint[1] = yExtent / 2. + gridDelta;
    }
    minPoint[D - 1] = base;
    maxPoint[D - 1] = base + height;

    viennals::MakeGeometry<NumericType, D> geo(
        cutout,
        SmartPointer<viennals::Box<NumericType, D>>::New(minPoint, maxPoint));
    geo.setIgnoreBoundaryConditions(true);
    geo.apply();
  }

  void getTaperedCutout(SmartPointer<viennals::Domain<NumericType, D>> cutout,
                        NumericType width, NumericType base, NumericType height,
                        NumericType angle) {
    auto mesh = SmartPointer<viennals::Mesh<NumericType>>::New();
    const NumericType offset = std::tan(angle * M_PI / 180.) * height;
    auto gridDelta = domain_->getGridDelta();
    if constexpr (D == 2) {
      for (int i = 0; i < 4; i++) {
        std::array<NumericType, 3> node = {0., 0., 0.};
        mesh->insertNextNode(node);
      }
      // x
      mesh->nodes[0][0] = -width / 2.;
      mesh->nodes[1][0] = width / 2.;
      mesh->nodes[2][0] = width / 2. + offset;
      mesh->nodes[3][0] = -width / 2. - offset;
      // y
      mesh->nodes[0][1] = base;
      mesh->nodes[1][1] = base;
      mesh->nodes[2][1] = base + height;
      mesh->nodes[3][1] = base + height;

      mesh->insertNextLine(std::array<unsigned, 2>{0, 3});
      mesh->insertNextLine(std::array<unsigned, 2>{3, 2});
      mesh->insertNextLine(std::array<unsigned, 2>{2, 1});
      mesh->insertNextLine(std::array<unsigned, 2>{1, 0});
    } else {
      auto gridDelta = domain_->getGridDelta();
      auto yExtent = domain_->getSetup().yExtent();
      for (int i = 0; i < 8; i++) {
        std::array<NumericType, 3> node = {0., 0., 0.};
        mesh->insertNextNode(node);
      }
      mesh->nodes[0][0] = -width / 2.;
      mesh->nodes[0][1] = -yExtent / 2. - gridDelta;
      mesh->nodes[0][2] = base;

      mesh->nodes[1][0] = width / 2.;
      mesh->nodes[1][1] = -yExtent / 2. - gridDelta;
      mesh->nodes[1][2] = base;

      mesh->nodes[2][0] = width / 2.;
      mesh->nodes[2][1] = yExtent / 2. + gridDelta;
      mesh->nodes[2][2] = base;

      mesh->nodes[3][0] = -width / 2.;
      mesh->nodes[3][1] = yExtent / 2. + gridDelta;
      mesh->nodes[3][2] = base;

      mesh->nodes[4][0] = -width / 2. - offset;
      mesh->nodes[4][1] = -yExtent / 2. - gridDelta;
      mesh->nodes[4][2] = height + base;

      mesh->nodes[5][0] = width / 2. + offset;
      mesh->nodes[5][1] = -yExtent / 2. - gridDelta;
      mesh->nodes[5][2] = height + base;

      mesh->nodes[6][0] = width / 2. + offset;
      mesh->nodes[6][1] = yExtent / 2. + gridDelta;
      mesh->nodes[6][2] = height + base;

      mesh->nodes[7][0] = -width / 2. - offset;
      mesh->nodes[7][1] = yExtent / 2. + gridDelta;
      mesh->nodes[7][2] = height + base;

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
    }
    viennals::FromSurfaceMesh<NumericType, D>(cutout, mesh).apply();
  }
};

} // namespace viennaps
