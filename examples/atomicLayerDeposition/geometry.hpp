#pragma once

#include <geometries/psMakeTrench.hpp>
#include <lsBooleanOperation.hpp>
#include <lsMakeGeometry.hpp>
#include <psDomain.hpp>
#include <psUtils.hpp>

using namespace viennacore;

template <class NumericType, int D>
class CustomSource : public viennaray::Source<NumericType, D> {
  const NumericType openingWidth2_;
  const NumericType height_;
  const NumericType gapWidth_;

public:
  CustomSource(NumericType openingWidth, NumericType height,
               NumericType gapWidth = 0.)
      : openingWidth2_(openingWidth / 2), height_(height), gapWidth_(gapWidth) {
  }

  Vec2D<Vec3D<NumericType>>
  getOriginAndDirection(const size_t idx, RNG &rngState) const override {
    std::uniform_real_distribution<NumericType> uniDist;

    NumericType x = uniDist(rngState) * openingWidth2_;
    Vec3D<NumericType> origin = {x., 0., 0.};
    origin[D - 1] = height_;
    if constexpr (D == 3) {
      origin[1] = uniDist(rngState) * gapWidth_ - gapWidth_ / 2.;
    }

    Vec3D<NumericType> direction;
    const NumericType tt = uniDist(rngState);
    const NumericType r1 = uniDist(rngState);
    direction[D - 1] = -std::sqrt(tt);
    direction[0] = std::cos(M_PI * 2. * r1) * std::sqrt(1 - tt);

    if constexpr (D == 2) {
      direction[2] = 0.;
      Normalize(direction);
    } else {
      direction[1] = std::sin(M_PI * 2. * r1) * std::sqrt(1 - tt);
    }

    return {origin, direction};
  }
  size_t getNumPoints() const override { return 1000; }
  NumericType getSourceArea() const override { return openingWidth2_; }
};

template <class NumericType, int D>
void makeHAR(SmartPointer<viennaps::Domain<NumericType, D>> domain,
             NumericType gridDelta, NumericType openingDepth,
             NumericType openingWidth, NumericType gapLength,
             NumericType gapHeight, NumericType xPad,
             viennaps::Material material) {
  //   static_assert(D == 2, "This function only works in 2D");
  domain->clear();

  double bounds[2 * D];
  bounds[0] = -xPad;
  bounds[1] = openingWidth + xPad + gapLength;

  bounds[2] = -gridDelta;
  bounds[3] = openingDepth + gapHeight + gridDelta;

  typename viennals::Domain<NumericType, D>::BoundaryType boundaryCons[D];

  for (int i = 0; i < D - 1; i++) {
    boundaryCons[i] =
        viennals::Domain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
  }
  boundaryCons[D - 1] =
      viennals::Domain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

  {
    auto substrate = SmartPointer<viennals::Domain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);
    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};
    normal[D - 1] = 1.;
    origin[D - 1] = openingDepth + gapHeight;
    viennals::MakeGeometry<NumericType, D>(
        substrate,
        SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(substrate, material);
  }

  {
    auto vertBox =
        SmartPointer<viennals::Domain<NumericType, D>>::New(domain->getGrid());
    NumericType minPoint[D] = {0., 0.};
    NumericType maxPoint[D] = {openingWidth,
                               gapHeight + openingDepth + gridDelta};
    viennals::MakeGeometry<NumericType, D>(
        vertBox,
        SmartPointer<viennals::Box<NumericType, D>>::New(minPoint, maxPoint))
        .apply();

    domain->applyBooleanOperation(
        vertBox, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
  }

  {
    auto horiBox =
        SmartPointer<viennals::Domain<NumericType, D>>::New(domain->getGrid());
    NumericType minPoint[D] = {openingWidth - gridDelta, 0.};
    NumericType maxPoint[D] = {openingWidth + gapLength, gapHeight};
    viennals::MakeGeometry<NumericType, D>(
        horiBox,
        SmartPointer<viennals::Box<NumericType, D>>::New(minPoint, maxPoint))
        .apply();

    domain->applyBooleanOperation(
        horiBox, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
  }
}

template <class NumericType, int D>
void makeT(SmartPointer<viennaps::Domain<NumericType, D>> domain,
           NumericType gridDelta, NumericType openingDepth,
           NumericType openingWidth, NumericType gapLength,
           NumericType gapHeight, NumericType xPad, viennaps::Material material,
           NumericType gapWidth = 0.) {
  //   static_assert(D == 2, "This function only works in 2D");
  domain->clear();
  typename viennals::Domain<NumericType, D>::BoundaryType boundaryCons[D];
  for (int i = 0; i < D - 1; i++) {
    boundaryCons[i] =
        viennals::Domain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
  }
  boundaryCons[D - 1] =
      viennals::Domain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

  if constexpr (D == 2) {
    double bounds[2 * D];
    bounds[0] = 0.;
    bounds[1] = openingWidth / 2. + xPad + gapLength;

    bounds[2] = -gridDelta;
    bounds[3] = openingDepth + gapHeight + gridDelta;

    {
      auto substrate = SmartPointer<viennals::Domain<NumericType, D>>::New(
          bounds, boundaryCons, gridDelta);
      NumericType normal[D] = {0.};
      NumericType origin[D] = {0.};
      normal[D - 1] = 1.;
      origin[D - 1] = openingDepth + gapHeight;
      viennals::MakeGeometry<NumericType, D>(
          substrate,
          SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
          .apply();
      domain->insertNextLevelSetAsMaterial(substrate, material);
    }

    {
      auto vertBox = SmartPointer<viennals::Domain<NumericType, D>>::New(
          domain->getGrid());
      NumericType minPoint[D] = {-gridDelta, 0.};
      NumericType maxPoint[D] = {openingWidth / 2.,
                                 gapHeight + openingDepth + gridDelta};
      viennals::MakeGeometry<NumericType, D>(
          vertBox,
          SmartPointer<viennals::Box<NumericType, D>>::New(minPoint, maxPoint))
          .apply();

      domain->applyBooleanOperation(
          vertBox, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
    }

    {
      auto horiBox = SmartPointer<viennals::Domain<NumericType, D>>::New(
          domain->getGrid());
      NumericType minPoint[D] = {openingWidth / 2. - gridDelta, 0.};
      NumericType maxPoint[D] = {openingWidth / 2. + gapLength, gapHeight};
      viennals::MakeGeometry<NumericType, D>(
          horiBox,
          SmartPointer<viennals::Box<NumericType, D>>::New(minPoint, maxPoint))
          .apply();

      domain->applyBooleanOperation(
          horiBox, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
    }
  } else {
    double bounds[2 * D];
    bounds[0] = 0.;
    bounds[1] = openingWidth / 2. + xPad + gapLength;

    bounds[2] = -gapWidth / 2. - xPad;
    bounds[3] = gapWidth / 2. + xPad;

    bounds[4] = -gridDelta;
    bounds[5] = openingDepth + gapHeight + gridDelta;

    {
      auto substrate = SmartPointer<viennals::Domain<NumericType, D>>::New(
          bounds, boundaryCons, gridDelta);
      NumericType normal[D] = {0.};
      NumericType origin[D] = {0.};
      normal[D - 1] = 1.;
      origin[D - 1] = openingDepth + gapHeight;
      viennals::MakeGeometry<NumericType, D>(
          substrate,
          SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
          .apply();
      domain->insertNextLevelSetAsMaterial(substrate, material);
    }

    {
      auto vertBox = SmartPointer<viennals::Domain<NumericType, D>>::New(
          domain->getGrid());
      NumericType minPoint[D] = {-gridDelta, -gapWidth / 2., 0.};
      NumericType maxPoint[D] = {openingWidth / 2., gapWidth / 2.,
                                 gapHeight + openingDepth + gridDelta};
      viennals::MakeGeometry<NumericType, D>(
          vertBox,
          SmartPointer<viennals::Box<NumericType, D>>::New(minPoint, maxPoint))
          .apply();

      domain->applyBooleanOperation(
          vertBox, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
    }

    {
      auto horiBox = SmartPointer<viennals::Domain<NumericType, D>>::New(
          domain->getGrid());
      NumericType minPoint[D] = {openingWidth / 2. - gridDelta, -gapWidth / 2.,
                                 0.};
      NumericType maxPoint[D] = {openingWidth / 2. + gapLength, gapWidth / 2.,
                                 gapHeight};
      viennals::MakeGeometry<NumericType, D>(
          horiBox,
          SmartPointer<viennals::Box<NumericType, D>>::New(minPoint, maxPoint))
          .apply();

      domain->applyBooleanOperation(
          horiBox, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
    }
  }
}