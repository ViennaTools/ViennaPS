#pragma once

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsMakeGeometry.hpp>

#include <psDomain.hpp>

/**
  Creates a trench geometry in z(3D)/y(2D) direction.
*/
template <class NumericType, int D> class psMakeTrench {
  using LSPtrType = psSmartPointer<lsDomain<NumericType, D>>;
  using PSPtrType = psSmartPointer<psDomain<NumericType, D>>;

public:
  PSPtrType domain = nullptr;

  NumericType gridDelta = .25;
  NumericType xExtent = 20;
  NumericType yExtent = 14;

  NumericType trenchWidth = 7;
  NumericType taperingAngle = 0;
  NumericType trenchDepth = 17.5;
  bool makeMask = true;

  psMakeTrench(PSPtrType passedDomain) : domain(passedDomain) {}

  psMakeTrench(PSPtrType passedDomain, const NumericType passedGridDelta,
               const NumericType passedXExtent, const NumericType passedYExtent,
               const NumericType passedTrenchWidth,
               const NumericType passedTrenchHeight,
               const bool passedMakeMask = true)
      : domain(passedDomain), gridDelta(passedGridDelta),
        xExtent(passedXExtent), yExtent(passedYExtent),
        trenchWidth(passedTrenchWidth), trenchDepth(passedTrenchHeight),
        makeMask(passedMakeMask) {}

  psMakeTrench(PSPtrType passedDomain, const NumericType passedGridDelta,
               const NumericType passedXExtent, const NumericType passedYExtent,
               const NumericType passedTrenchWidth,
               const NumericType passedTrenchHeight,
               const NumericType passedTaperingAngle,
               const bool passedMakeMask = true)
      : domain(passedDomain), gridDelta(passedGridDelta),
        xExtent(passedXExtent), yExtent(passedYExtent),
        trenchWidth(passedTrenchWidth), taperingAngle(passedTaperingAngle),
        trenchDepth(passedTrenchHeight), makeMask(passedMakeMask) {}

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

    for (int i = 0; i < D - 1; i++)
      boundaryCons[i] =
          lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[D - 1] =
        lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    auto substrate = LSPtrType::New(bounds, boundaryCons, gridDelta);
    NumericType normal[D] = {0.};
    NumericType origin[D] = {0.};
    normal[D - 1] = 1.;
    lsMakeGeometry<NumericType, D>(
        substrate, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    auto mask = LSPtrType::New(bounds, boundaryCons, gridDelta);
    origin[D - 1] = trenchDepth;
    lsMakeGeometry<NumericType, D>(
        mask, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    auto maskAdd = LSPtrType::New(bounds, boundaryCons, gridDelta);
    origin[D - 1] = 0.;
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
        mesh->nodes[2][1] = trenchDepth;
        mesh->nodes[3][0] = -trenchWidth / 2. - offset;
        mesh->nodes[3][1] = trenchDepth;

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
        mesh->nodes[0][1] = -yExtent / 2.;

        mesh->nodes[1][0] = trenchWidth / 2.;
        mesh->nodes[1][1] = -yExtent / 2.;

        mesh->nodes[2][0] = trenchWidth / 2.;
        mesh->nodes[2][1] = yExtent / 2.;

        mesh->nodes[3][0] = -trenchWidth / 2.;
        mesh->nodes[3][1] = yExtent / 2.;

        mesh->nodes[4][0] = -trenchWidth / 2. - offset;
        mesh->nodes[4][1] = -yExtent / 2.;
        mesh->nodes[4][2] = trenchDepth;

        mesh->nodes[5][0] = trenchWidth / 2. + offset;
        mesh->nodes[5][1] = -yExtent / 2.;
        mesh->nodes[5][2] = trenchDepth;

        mesh->nodes[6][0] = trenchWidth / 2. + offset;
        mesh->nodes[6][1] = yExtent / 2.;
        mesh->nodes[6][2] = trenchDepth;

        mesh->nodes[7][0] = -trenchWidth / 2. - offset;
        mesh->nodes[7][1] = yExtent / 2.;
        mesh->nodes[7][2] = trenchDepth;

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
        minPoint[2] = 0.;
        maxPoint[2] = trenchDepth;
      } else {
        minPoint[1] = 0.;
        maxPoint[1] = trenchDepth;
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