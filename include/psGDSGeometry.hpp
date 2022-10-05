#pragma once

#include <lsBooleanOperation.hpp>
#include <lsConvexHull.hpp>
#include <lsDomain.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsGeometries.hpp>
#include <lsVTKWriter.hpp>

#include <psGDSUtils.hpp>
#include <psSmartPointer.hpp>

template <class NumericType, int D>
void printLS(psSmartPointer<lsDomain<NumericType, D>> domain,
             std::string name) {
  auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
  lsToSurfaceMesh<NumericType, D>(domain, mesh).apply();
  lsVTKWriter<NumericType>(mesh, name).apply();
}

template <class NumericType, int D = 3> class psGDSGeometry {
  std::vector<psGDSStructure<NumericType>> structures;
  std::unordered_map<int16_t, psSmartPointer<lsDomain<NumericType, 3>>>
      assembledStructures;
  std::string libName = "";
  double units;
  double userUnits;
  std::array<NumericType, 2> boundaryPadding = {0., 0.};
  std::array<NumericType, 2> minBounds;
  std::array<NumericType, 2> maxBounds;

public:
  psGDSGeometry() {}

  void setLibName(const char *str) { libName = str; }

  void setBoundaryPadding(const NumericType xPadding,
                          const NumericType yPadding) {
    boundaryPadding[0] = xPadding;
    boundaryPadding[1] = yPadding;
  }

  void insertNextStructure(psGDSStructure<NumericType> &structure) {
    structures.push_back(structure);
  }

  void print() {
    std::cout << "======= STRUCTURES ========" << std::endl;
    for (auto &s : structures) {
      s.print();
    }
    std::cout << "============================" << std::endl;
  }

  psGDSStructure<NumericType> *getStructure(std::string strName) {
    for (size_t i = 0; i < structures.size(); i++) {
      if (strName == structures[i].name) {
        return &structures[i];
      }
    }
    return nullptr;
  }

  psSmartPointer<lsDomain<NumericType, D>>
  layerToLevelSet(const int16_t layer, const NumericType height,
                  const NumericType gridDelta, bool mask = false) {
    double bounds[2 * D] = {minBounds[0] - boundaryPadding[0],
                            maxBounds[0] + boundaryPadding[0],
                            minBounds[1] - boundaryPadding[1],
                            maxBounds[1] + boundaryPadding[1],
                            -1.,
                            1.};

    typename lsDomain<NumericType, D>::BoundaryType boundaryCons[3] = {
        lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY,
        lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY,
        lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY};

    auto levelSet = psSmartPointer<lsDomain<NumericType, D>>::New(
        bounds, boundaryCons, gridDelta);

    for (auto &str : structures) {
      if (!str.isRef) {
        for (auto &el : str.elements) {
          if (el.layer == layer) {
            if (el.elementType == elBox) {
              addBox(levelSet, el, height, 0., 0.);
            } else {
              addPolygon(levelSet, el, height, 0, 0);
            }
          }
        }

        for (auto &sref : str.sRefs) {
          auto refStr = getStructure(sref.strName);
          for (auto &el : refStr->elements) {
            if (el.layer == layer) {
              if (el.elementType == elBox) {
                addBox(levelSet, el, height, sref.refPoint[0],
                       sref.refPoint[1]);
              } else {
                addPolygon(levelSet, el, height, sref.refPoint[0],
                           sref.refPoint[1]);
              }
            }
          }
        }
      }
    }

    if (mask) {
      auto topPlane = psSmartPointer<lsDomain<NumericType, D>>::New(
          bounds, boundaryCons, gridDelta);
      NumericType normal[D] = {0., 0., 1.};
      NumericType origin[D] = {0., 0., height};
      lsMakeGeometry<NumericType, D>(
          topPlane,
          lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
          .apply();

      auto botPlane = psSmartPointer<lsDomain<NumericType, D>>::New(
          bounds, boundaryCons, gridDelta);
      normal[D - 1] = -1.;
      origin[D - 1] = 0.;
      lsMakeGeometry<NumericType, D>(
          botPlane,
          lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
          .apply();

      lsBooleanOperation<NumericType, D>(topPlane, botPlane,
                                         lsBooleanOperationEnum::INTERSECT)
          .apply();

      lsBooleanOperation<NumericType, D>(
          topPlane, levelSet, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();

      return topPlane;
    }
    return levelSet;
  }

  void checkReferences() {
    for (auto &str : structures) {
      for (auto &sref : str.sRefs) {
        auto refStr = getStructure(sref.strName);
        refStr->isRef = true;
      }
    }
  }

  void calculateBoundingBoxes() {
    minBounds[0] = std::numeric_limits<NumericType>::max();
    minBounds[1] = std::numeric_limits<NumericType>::max();

    maxBounds[0] = std::numeric_limits<NumericType>::lowest();
    maxBounds[1] = std::numeric_limits<NumericType>::lowest();

    size_t numStructures = structures.size();
    std::unordered_map<std::string, bool> processed;
    for (const auto &str : structures) {
      processed.insert({str.name, false});
    }

    while (!std::all_of(processed.begin(), processed.end(),
                        [](std::pair<const std::basic_string<char>, bool> &p) {
                          return p.second;
                        })) {
      for (size_t i = 0; i < numStructures; i++) {
        if (processed[structures[i].name])
          continue;

        structures[i].boundingBox = structures[i].elementBoundingBox;
        bool finish = true;
        for (const auto sref : structures[i].sRefs) {
          auto refStr = getStructure(sref.strName);
          assert(refStr);
          if (!processed[refStr->name]) {
            finish = false;
            break;
          }

          auto minPoint_x = sref.refPoint[0] - refStr->boundingBox[0][0];
          auto minPoint_y = sref.refPoint[1] - refStr->boundingBox[0][1];

          auto maxPoint_x = sref.refPoint[0] + refStr->boundingBox[1][0];
          auto maxPoint_y = sref.refPoint[1] + refStr->boundingBox[1][1];

          if (minPoint_x < structures[i].boundingBox[0][0]) {
            structures[i].boundingBox[0][0] = minPoint_x;
          }
          if (minPoint_y < structures[i].boundingBox[0][1]) {
            structures[i].boundingBox[0][1] = minPoint_y;
          }
          if (maxPoint_x > structures[i].boundingBox[1][0]) {
            structures[i].boundingBox[1][0] = maxPoint_x;
          }
          if (maxPoint_y > structures[i].boundingBox[1][1]) {
            structures[i].boundingBox[1][1] = maxPoint_y;
          }
        }
        if (!finish)
          continue;

        if (!structures[i].isRef) {
          if (structures[i].boundingBox[0][0] < minBounds[0]) {
            minBounds[0] = structures[i].boundingBox[0][0];
          }
          if (structures[i].boundingBox[0][1] < minBounds[1]) {
            minBounds[1] = structures[i].boundingBox[0][1];
          }
          if (structures[i].boundingBox[1][0] > maxBounds[0]) {
            maxBounds[0] = structures[i].boundingBox[1][0];
          }
          if (structures[i].boundingBox[1][1] > maxBounds[1]) {
            maxBounds[1] = structures[i].boundingBox[1][1];
          }
        }

        processed[structures[i].name] = true;
      }
    }
  }

  void printBound() const {
    std::cout << "Geometry: (" << minBounds[0] << ", " << minBounds[1]
              << ") - (" << maxBounds[0] << ", " << maxBounds[1] << ")"
              << std::endl;
  }

  void addBox(psSmartPointer<lsDomain<NumericType, D>> levelSet,
              psGDSElement<NumericType> &element, const NumericType height,
              const NumericType xOffset, const NumericType yOffset) {
    auto tmpLS =
        psSmartPointer<lsDomain<NumericType, D>>::New(levelSet->getGrid());

    auto minPoint = element.pointCloud[1];
    minPoint[2] = 0.;
    auto maxPoint = element.pointCloud[3];
    maxPoint[2] = height;

    lsMakeGeometry<NumericType, D>(
        tmpLS, lsSmartPointer<lsBox<NumericType, D>>::New(minPoint.data(),
                                                          maxPoint.data()))
        .apply();
    lsBooleanOperation<NumericType, D>(levelSet, tmpLS,
                                       lsBooleanOperationEnum::UNION)
        .apply();
  }

  void addPolygon(psSmartPointer<lsDomain<NumericType, D>> levelSet,
                  psGDSElement<NumericType> &element, const NumericType height,
                  const NumericType xOffset, const NumericType yOffset) {
    auto mesh = elementToSurfaceMesh(element, height, xOffset, yOffset);
    auto tmpLS =
        psSmartPointer<lsDomain<NumericType, D>>::New(levelSet->getGrid());
    lsFromSurfaceMesh<NumericType, D>(tmpLS, mesh).apply();
    lsBooleanOperation<NumericType, D>(levelSet, tmpLS,
                                       lsBooleanOperationEnum::UNION)
        .apply();
  }

  psSmartPointer<lsMesh<NumericType>>
  elementToSurfaceMesh(psGDSElement<NumericType> &element,
                       const NumericType height, const NumericType xOffset,
                       const NumericType yOffset) {
    auto mesh = psSmartPointer<lsMesh<NumericType>>::New();

    unsigned numPointsFlat = element.pointCloud.size();

    // sidewalls
    for (unsigned i = 0; i < numPointsFlat; i++) {
      std::array<NumericType, D> offsetPoint = element.pointCloud[i];
      offsetPoint[0] += xOffset;
      offsetPoint[1] += yOffset;
      mesh->insertNextNode(offsetPoint);

      mesh->insertNextTriangle(std::array<unsigned, 3>{
          i, (i + 1) % numPointsFlat, i + numPointsFlat});
    }

    for (unsigned i = 0; i < numPointsFlat; i++) {
      unsigned upPoint = i + numPointsFlat;
      std::array<NumericType, D> offsetPoint = element.pointCloud[i];
      offsetPoint[0] += xOffset;
      offsetPoint[1] += yOffset;
      offsetPoint[2] = height;
      mesh->insertNextNode(offsetPoint);

      mesh->insertNextTriangle(std::array<unsigned, 3>{
          upPoint, (upPoint + 1) % numPointsFlat,
          (upPoint + 1) % numPointsFlat + numPointsFlat});
    }

    // polygon triangulation (ear clipping algorithm)
    std::vector<unsigned> leftNeighbors(numPointsFlat);
    std::vector<unsigned> rightNeighbors(numPointsFlat);

    // initialize neighbors
    for (unsigned i = 0; i < numPointsFlat; i++) {
      leftNeighbors[i] = ((i - 1) + numPointsFlat) % numPointsFlat;
      rightNeighbors[i] = ((i + 1) + numPointsFlat) % numPointsFlat;
    }

    unsigned numTriangles = 0;
    unsigned i = numPointsFlat - 1;
    while (numTriangles < (numPointsFlat - 2)) {
      i = rightNeighbors[i];
      if (isEar(leftNeighbors[i], i, rightNeighbors[i], mesh, numPointsFlat)) {
        mesh->insertNextTriangle(
            std::array<unsigned, 3>{rightNeighbors[i], i, leftNeighbors[i]});

        // remove point
        leftNeighbors[rightNeighbors[i]] = leftNeighbors[i];
        rightNeighbors[leftNeighbors[i]] = rightNeighbors[i];

        numTriangles++;
      }
    }

    // use same triangles for other side
    auto &triangles = mesh->template getElements<3>();
    unsigned numPrevTriangles = triangles.size() - 1;
    for (unsigned j = 0; j < numTriangles; j++) {
      auto triangle = triangles[numPrevTriangles - j];
      for (int d = 0; d < 3; d++) {
        triangle[d] += numPointsFlat;
      }
      swap(triangle[0], triangle[2]); // swap for correct orientation
      mesh->insertNextTriangle(triangle);
    }

    return mesh;
  }

  bool isEar(int i, int j, int k, psSmartPointer<lsMesh<NumericType>> mesh,
             unsigned numPoints) {
    auto &points = mesh->getNodes();

    // check if triangle is clockwise orientated
    if ((points[i][0] * points[j][1] + points[i][1] * points[k][0] +
         points[j][0] * points[k][1] - points[k][0] * points[j][1] -
         points[k][1] * points[i][0] - points[j][0] * points[i][1]) < 0.)
      return false;

    for (unsigned m = 0; m < numPoints; m++) {
      if ((m != i) && (m != j) && (m != k)) {
        // check if point in triangle
        auto side_1 =
            (points[m][0] - points[j][0]) * (points[i][1] - points[j][1]) -
            (points[i][0] - points[j][0]) * (points[m][1] - points[j][1]);
        // Segment B to C
        auto side_2 =
            (points[m][0] - points[k][0]) * (points[j][1] - points[k][1]) -
            (points[j][0] - points[k][0]) * (points[m][1] - points[k][1]);
        // Segment C to A
        auto side_3 =
            (points[m][0] - points[i][0]) * (points[k][1] - points[i][1]) -
            (points[k][0] - points[i][0]) * (points[m][1] - points[i][1]);

        // All the signs must be positive or all negative
        if (((side_1 < 0.0) && (side_2 < 0.0) && (side_3 < 0.0)) ||
            ((side_1 > 0.0) && (side_2 > 0.0) && (side_3 > 0.0)))
          return false;
      }
    }

    return true;
  }

  inline void swap(unsigned &a, unsigned &b) {
    unsigned t = a;
    a = b;
    b = t;
  }
};