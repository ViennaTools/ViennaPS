#pragma once

#include <lsBooleanOperation.hpp>
#include <lsConvexHull.hpp>
#include <lsDomain.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsGeometries.hpp>
#include <lsVTKWriter.hpp>

#include <psGDSUtils.hpp>

template <class NumericType, int D = 3> class psGDSGeometry {
  std::vector<psGDSStructure<NumericType>> structures;
  std::string libName = "";
  double units;
  double userUnits;
  std::array<NumericType, 2> boundaryPadding = {0., 0.};
  std::array<NumericType, 2> minBounds;
  std::array<NumericType, 2> maxBounds;

public:
  psGDSGeometry() {}

  void setLibName(const char *str) { libName = str; }

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

  psGDSStructure<NumericType> *getStructure(std::string strName) const {
    for (size_t i = 0; i < structures.size(); i++) {
      if (strName == structures[i].name) {
        return &structures[i];
      }
      return nullptr;
    }
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
      for (auto &el : str.elements) {
        if (el.layer == layer) {

          auto mesh = elementToSurfaceMesh(el, height);

          auto tmpLS = psSmartPointer<lsDomain<NumericType, D>>::New(
              bounds, boundaryCons, gridDelta);
          lsFromSurfaceMesh<NumericType, D>(tmpLS, mesh).apply();
          lsBooleanOperation<NumericType, D>(levelSet, tmpLS,
                                             lsBooleanOperationEnum::UNION)
              .apply();
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

  void calculateBoundingBoxes() {
    minBounds = structures[0].elementBoundingBox[0];
    maxBounds = structures[0].elementBoundingBox[1];

    for (auto &str : structures) {
      str.boundingBox = str.elementBoundingBox;
      // TODO
      //   for (const auto sref : str.sRefs) {
      //   }

      if (str.boundingBox[0][0] < minBounds[0]) {
        minBounds[0] = str.boundingBox[0][0];
      }
      if (str.boundingBox[0][1] < minBounds[1]) {
        minBounds[1] = str.boundingBox[0][1];
      }
      if (str.boundingBox[1][0] < maxBounds[0]) {
        maxBounds[0] = str.boundingBox[1][0];
      }
      if (str.boundingBox[1][1] < maxBounds[1]) {
        maxBounds[1] = str.boundingBox[1][1];
      }
    }
  }

  psSmartPointer<lsMesh<NumericType>>
  elementToSurfaceMesh(psGDSElement<NumericType> &element,
                       const NumericType height) {
    auto mesh = psSmartPointer<lsMesh<NumericType>>::New();

    unsigned numPointsFlat = element.pointCloud.size();
    for (auto p : element.pointCloud) {
      std::array<NumericType, D> point = p;
      point[2] = height;
      element.pointCloud.push_back(point);
    }

    // sidewalls
    for (unsigned i = 0; i < numPointsFlat; i++) {
      mesh->insertNextNode(element.pointCloud[i]);

      mesh->insertNextTriangle(std::array<unsigned, 3>{
          i, (i + 1) % numPointsFlat, i + numPointsFlat});
    }

    for (unsigned i = 0; i < numPointsFlat; i++) {
      unsigned upPoint = i + numPointsFlat;
      mesh->insertNextNode(element.pointCloud[upPoint]);

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