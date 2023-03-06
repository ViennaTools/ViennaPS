#pragma once

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsMakeGeometry.hpp>

#include <psDomain.hpp>

/**
 * Creates a trench geometry in z(3D)/y(2D) direction.
 */
template <class NumericType, int D> class psMakeTrench {
  using LSPtrType = psSmartPointer<lsDomain<NumericType, D>>;
  using PSPtrType = psSmartPointer<psDomain<NumericType, D>>;

public:
  PSPtrType domain = nullptr;

  NumericType gridDelta = .25;
  NumericType xExtent = 20;
  NumericType yExtent = 14;
  NumericType baseHeight = 0.;

  NumericType trenchWidth = 7;
  NumericType taperingAngle = 0;
  NumericType trenchDepth = 17.5;
  bool periodicBoundary = false;
  bool makeMask = false;

  psMakeTrench(PSPtrType passedDomain, const NumericType passedGridDelta,
               const NumericType passedXExtent, const NumericType passedYExtent,
               const NumericType passedTrenchWidth,
               const NumericType passedTrenchHeight,
               const NumericType passedTaperingAngle = 0.,
               const NumericType passedBaseHeight = 0.,
               const bool passedPeriodicBoundary = false,
               const bool passedMakeMask = false)
      : domain(passedDomain), gridDelta(passedGridDelta),
        xExtent(passedXExtent), yExtent(passedYExtent),
        trenchWidth(passedTrenchWidth), trenchDepth(passedTrenchHeight),
        taperingAngle(passedTaperingAngle), baseHeight(passedBaseHeight),
        periodicBoundary(passedPeriodicBoundary), makeMask(passedMakeMask) {}

  void apply() {
    domain->clear();
    double bounds[2 * D];
    bounds[0] = -xExtent / 2.;
    bounds[1] = xExtent / 2.;

    if constexpr (D == 3) {
      bounds[2] = -yExtent / 2.;
      bounds[3] = yExtent / 2.;
      bounds[4] = -gridDelta;
      bounds[5] = trenchDepth + gridDelta;
    } else {
      bounds[2] = -gridDelta;
      bounds[3] = trenchDepth + gridDelta;
    }

    typename lsDomain<NumericType, D>::BoundaryType boundaryCons[D];

    for (int i = 0; i < D - 1; i++) {
      if (periodicBoundary) {
        boundaryCons[i] =
            lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY;
      } else {
        boundaryCons[i] =
            lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
      }
    }
    boundaryCons[D - 1] =
        lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    auto substrate = LSPtrType::New(bounds, boundaryCons, gridDelta);
    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};
    normal[D - 1] = 1.;
    origin[D - 1] = baseHeight;
    lsMakeGeometry<NumericType, D>(
        substrate, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    auto mask = LSPtrType::New(bounds, boundaryCons, gridDelta);
    origin[D - 1] = trenchDepth + baseHeight;
    lsMakeGeometry<NumericType, D>(
        mask, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    auto maskAdd = LSPtrType::New(bounds, boundaryCons, gridDelta);
    origin[D - 1] = baseHeight;
    normal[D - 1] = -1.;
    lsMakeGeometry<NumericType, D>(
        maskAdd, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    lsBooleanOperation<NumericType, D>(mask, maskAdd,
                                       lsBooleanOperationEnum::INTERSECT)
        .apply();

    auto cutout = LSPtrType::New(bounds, boundaryCons, gridDelta);

    if (taperingAngle) {
      auto mesh = psSmartPointer<lsMesh<NumericType>>::New();
      const NumericType offset =
          std::tan(taperingAngle * rayInternal::PI / 180.) * trenchDepth;
      if constexpr (D == 2) {
        for (int i = 0; i < 4; i++) {
          std::array<NumericType, 3> node = {0., 0., 0.};
          mesh->insertNextNode(node);
        }
        mesh->nodes[0][0] = -trenchWidth / 2.;
        mesh->nodes[1][0] = trenchWidth / 2.;
        mesh->nodes[2][0] = trenchWidth / 2. + offset;
        mesh->nodes[3][0] = -trenchWidth / 2. - offset;

        mesh->nodes[0][1] = baseHeight;
        mesh->nodes[1][1] = baseHeight;
        mesh->nodes[2][1] = trenchDepth + baseHeight;
        mesh->nodes[3][1] = trenchDepth + baseHeight;

        mesh->insertNextLine(std::array<unsigned, 2>{0, 3});
        mesh->insertNextLine(std::array<unsigned, 2>{3, 2});
        mesh->insertNextLine(std::array<unsigned, 2>{2, 1});
        mesh->insertNextLine(std::array<unsigned, 2>{1, 0});
        lsFromSurfaceMesh<NumericType, D>(cutout, mesh).apply();
      } else {
        for (int i = 0; i < 8; i++) {
          std::array<NumericType, 3> node = {0., 0., 0.};
          mesh->insertNextNode(node);
        }
        mesh->nodes[0][0] = -trenchWidth / 2.;
        mesh->nodes[0][1] = -yExtent / 2. - gridDelta;
        mesh->nodes[0][2] = baseHeight;

        mesh->nodes[1][0] = trenchWidth / 2.;
        mesh->nodes[1][1] = -yExtent / 2. - gridDelta;
        mesh->nodes[1][2] = baseHeight;

        mesh->nodes[2][0] = trenchWidth / 2.;
        mesh->nodes[2][1] = yExtent / 2. + gridDelta;
        mesh->nodes[2][2] = baseHeight;

        mesh->nodes[3][0] = -trenchWidth / 2.;
        mesh->nodes[3][1] = yExtent / 2. + gridDelta;
        mesh->nodes[3][2] = baseHeight;

        mesh->nodes[4][0] = -trenchWidth / 2. - offset;
        mesh->nodes[4][1] = -yExtent / 2. - gridDelta;
        mesh->nodes[4][2] = trenchDepth + baseHeight;

        mesh->nodes[5][0] = trenchWidth / 2. + offset;
        mesh->nodes[5][1] = -yExtent / 2. - gridDelta;
        mesh->nodes[5][2] = trenchDepth + baseHeight;

        mesh->nodes[6][0] = trenchWidth / 2. + offset;
        mesh->nodes[6][1] = yExtent / 2. + gridDelta;
        mesh->nodes[6][2] = trenchDepth + baseHeight;

        mesh->nodes[7][0] = -trenchWidth / 2. - offset;
        mesh->nodes[7][1] = yExtent / 2. + gridDelta;
        mesh->nodes[7][2] = trenchDepth + baseHeight;

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

        lsFromSurfaceMesh<NumericType, D>(cutout, mesh).apply();
      }
    } else {
      NumericType minPoint[D];
      NumericType maxPoint[D];

      minPoint[0] = -trenchWidth / 2;
      maxPoint[0] = trenchWidth / 2;

      if constexpr (D == 3) {
        minPoint[1] = -yExtent / 2.;
        maxPoint[1] = yExtent / 2.;
        minPoint[2] = baseHeight;
        maxPoint[2] = trenchDepth + baseHeight;
      } else {
        minPoint[1] = baseHeight;
        maxPoint[1] = trenchDepth + baseHeight;
      }
      lsMakeGeometry<NumericType, D>(
          cutout,
          lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
          .apply();
    }

    lsBooleanOperation<NumericType, D>(
        mask, cutout, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
        .apply();

    lsBooleanOperation<NumericType, D>(substrate, mask,
                                       lsBooleanOperationEnum::UNION)
        .apply();

    if (makeMask)
      domain->insertNextLevelSet(mask);
    domain->insertNextLevelSet(substrate, false);
  }
};