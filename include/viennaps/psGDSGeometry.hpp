#pragma once

#include "psGDSUtils.hpp"
#include "psGDSMaskProximity.hpp"

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsGeometries.hpp>
#include <lsMakeGeometry.hpp>
#include <lsTransformMesh.hpp>

#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>
#include <cmath>

namespace viennaps {

using namespace viennacore;

template <class NumericType, int D = 3> class GDSGeometry {
  using StructureLayers =
      std::unordered_map<int16_t, SmartPointer<viennals::Mesh<NumericType>>>;
  using lsDomainType = SmartPointer<viennals::Domain<NumericType, D>>;
  using BoundaryType = typename viennals::Domain<NumericType, D>::BoundaryType;

public:
  GDSGeometry() {
    if constexpr (D == 2) {
      Logger::getInstance()
          .addError("Cannot import 2D geometry from GDS file.")
          .print();
    }
  }

  explicit GDSGeometry(const NumericType gridDelta) : gridDelta_(gridDelta) {
    if constexpr (D == 2) {
      Logger::getInstance()
          .addError("Cannot import 2D geometry from GDS file.")
          .print();
    }
  }

  void setGridDelta(const NumericType gridDelta) { gridDelta_ = gridDelta; }

  void setBoundaryPadding(const NumericType xPadding,
                          const NumericType yPadding) {
    boundaryPadding[0] = xPadding;
    boundaryPadding[1] = yPadding;
  }

  void setBoundaryConditions(BoundaryType boundaryConds[3]) {
    for (int i = 0; i < 3; i++)
      boundaryConds_[i] = boundaryConds[i];
  }

  void print() const {
    std::cout << "======== STRUCTURES ========" << std::endl;
    for (auto &s : structures) {
      s.print();
    }
    std::cout << "============================" << std::endl;
  }

  lsDomainType layerToLevelSet(const int16_t layer,
                               const NumericType baseHeight,
                               const NumericType height, bool mask = false) {

    auto levelSet = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);

    for (auto &str : structures) { // loop over all structures
      if (!str.isRef) {
        // add single elements
        if (auto contains = str.containsLayers.find(layer);
            contains != str.containsLayers.end()) {

          // Step 1: Collect all polygons across the entire layer
          std::vector<std::vector<std::pair<double, double>>> allPolygons;
          for (auto &el : str.elements) {
            if (el.layer == layer) {
              std::vector<std::pair<double, double>> polygon;
              for (const auto &point : el.pointCloud) {
                polygon.push_back(std::pair<double, double>(point[0], point[1]));
              }
              allPolygons.push_back(polygon);
            }
          }

          // Step 2: Apply Gaussian convolution to simulate global proximity effects
          double polygonMaxX = 70.0;
          double polygonMaxY = 70.0;
          double proximitySigma = 0.01;
          psGDSMaskProximity<NumericType> proximityEffect(proximitySigma, polygonMaxX, polygonMaxY);
          proximityEffect.addPolygons(allPolygons);

          double forwardSigma = 3.0;
          double backwardSigma = 50.0;
          proximityEffect.applyProximityEffects(forwardSigma, backwardSigma);

          // Step 3: Extract modified exposure contours at threshold 0.5
          std::vector<std::vector<std::pair<double, double>>> modifiedPolygons = 
              proximityEffect.extractContoursAtThreshold(0.5);

          int i = 0;
          auto elementIt = str.elements.begin();  // Use iterator to avoid reallocation issues

          while (elementIt != str.elements.end() && !modifiedPolygons.empty()) {
              if (elementIt->layer == layer) {
                  elementIt->pointCloud.clear();

                  // Assign polygon to current element
                  std::vector<std::pair<double, double>> simplifiedPolygon = simplifyPolygon(modifiedPolygons.front(), 0.10);
                  proximityEffect.saveContoursToCSV(simplifiedPolygon, "simple_polygon_" + std::to_string(i) + ".csv");
                  i++;

                  // Compute scaling factors
                  double scaleX = (bounds_[1] - bounds_[0]) / (polygonMaxX);
                  double scaleY = (bounds_[3] - bounds_[2]) / (polygonMaxY);

                  for (const auto &point : simplifiedPolygon) {
                      double scaledX = (point.first) * scaleX + bounds_[0];
                      double scaledY = (point.second) * scaleY + bounds_[2];

                      elementIt->pointCloud.push_back({scaledX, scaledY, 0.});
                  }

                  if (elementIt->elementType == GDS::ElementType::elBox) {
                    addBox(levelSet, *elementIt, baseHeight, height, 0., 0.);
                  } else {
                    addPolygon(levelSet, *elementIt, baseHeight, height, 0., 0.);
                  }
                  // Remove the processed polygon
                  modifiedPolygons.erase(modifiedPolygons.begin());
              }

              ++elementIt;
          }

          // Remove leftover empty elements (only from the specified layer)
          str.elements.erase(std::remove_if(str.elements.begin(), str.elements.end(),
                                            [&](const auto &el) {
                                                return el.layer == layer && el.pointCloud.empty();
                                            }),
                            str.elements.end());

        }

        // add structure references
        auto strMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
        for (auto &sref : str.sRefs) {
          auto refStr = getStructure(sref.strName);
          if (auto contains = refStr->containsLayers.find(layer);
              contains != refStr->containsLayers.end()) {
            assert(assembledStructures[refStr->name][layer]);

            // copy mesh here
            auto copy = assembledStructures[refStr->name][layer];
            auto preBuiltStrMesh =
                SmartPointer<viennals::Mesh<NumericType>>::New();
            preBuiltStrMesh->nodes = copy->nodes;
            preBuiltStrMesh->triangles = copy->triangles;
            adjustPreBuiltMeshHeight(preBuiltStrMesh, baseHeight, height);

            if (sref.angle > 0.) {
              viennals::TransformMesh<NumericType>(
                  preBuiltStrMesh, viennals::TransformEnum::ROTATION,
                  hrleVectorType<double, 3>{0., 0., 1.}, deg2rad(sref.angle))
                  .apply();
            }

            if (sref.magnification > 0.) {
              viennals::TransformMesh<NumericType>(
                  preBuiltStrMesh, viennals::TransformEnum::SCALE,
                  hrleVectorType<double, 3>{sref.magnification,
                                            sref.magnification, 1.})
                  .apply();
            }

            if (sref.flipped) {
              Logger::getInstance()
                  .addWarning("Flipping x-axis currently not supported.")
                  .print();
              continue;
            }

            viennals::TransformMesh<NumericType>(
                preBuiltStrMesh, viennals::TransformEnum::TRANSLATION,
                hrleVectorType<double, 3>{sref.refPoint[0], sref.refPoint[1],
                                          0.})
                .apply();

            strMesh->append(*preBuiltStrMesh);
          }
        }
        if (strMesh->nodes.size() > 0) {
          auto tmpLS = lsDomainType::New(levelSet->getGrid());
          viennals::FromSurfaceMesh<NumericType, D>(tmpLS, strMesh).apply();
          viennals::BooleanOperation<NumericType, D>(
              levelSet, tmpLS, viennals::BooleanOperationEnum::UNION)
              .apply();
        }
      }
    }

    if (mask) {
      auto topPlane = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
      NumericType normal[3] = {0., 0., 1.};
      NumericType origin[3] = {0., 0., baseHeight + height};
      viennals::MakeGeometry<NumericType, D>(
          topPlane,
          SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
          .apply();

      auto botPlane = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
      normal[D - 1] = -1.;
      origin[D - 1] = baseHeight;
      viennals::MakeGeometry<NumericType, D>(
          botPlane,
          SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
          .apply();

      viennals::BooleanOperation<NumericType, D>(
          topPlane, botPlane, viennals::BooleanOperationEnum::INTERSECT)
          .apply();

      viennals::BooleanOperation<NumericType, D>(
          topPlane, levelSet,
          viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();

      return topPlane;
    }
    return levelSet;
  }

  void printBound() const {
    std::cout << "Geometry: (" << minBounds[0] << ", " << minBounds[1]
              << ") - (" << maxBounds[0] << ", " << maxBounds[1] << ")"
              << std::endl;
  }

  std::array<std::array<NumericType, 2>, 2> getBoundingBox() const {
    return {minBounds, maxBounds};
  }

  auto getBounds() { return bounds_; }

  void insertNextStructure(GDS::Structure<NumericType> const &structure) {
    structures.push_back(structure);
  }

  void finalize() {
    checkReferences();
    preBuildStructures();
    calculateBoundingBoxes();
  }

private:

  // Function to compute angle change between three points
  double computeAngle(std::pair<double, double> p1, 
                      std::pair<double, double> p2, 
                      std::pair<double, double> p3) {
      double v1x = p2.first - p1.first;
      double v1y = p2.second - p1.second;
      double v2x = p3.first - p2.first;
      double v2y = p3.second - p2.second;

      double dotProduct = v1x * v2x + v1y * v2y;
      double magnitude1 = std::sqrt(v1x * v1x + v1y * v1y);
      double magnitude2 = std::sqrt(v2x * v2x + v2y * v2y);

      if (magnitude1 == 0.0 || magnitude2 == 0.0) return 0.0; // Avoid division by zero

      double cosTheta = dotProduct / (magnitude1 * magnitude2);
      return std::acos(std::clamp(cosTheta, -1.0, 1.0)); // Clamp to avoid NaN
  }

  // Function to compute Euclidean distance
  double euclideanDistance(const std::pair<double, double>& p1, const std::pair<double, double>& p2) {
    return std::sqrt(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2));
  }

  // Function to find the two adjacent points in the polygon that are farthest apart
  size_t findFarthestAdjacentPoints(const std::vector<std::pair<double, double>>& polygon) {
    if (polygon.size() < 2) return 0; // Not enough points to process

    size_t maxIdx = 0;
    double maxDistance = 0.0;

    for (size_t i = 0; i < polygon.size() - 1; ++i) {
        double dist = euclideanDistance(polygon[i], polygon[i + 1]);
        if (dist > maxDistance) {
            maxDistance = dist;
            maxIdx = i;
        }
    }
    return maxIdx;  // Return index of the first point in the farthest pair
  }

  // Function to simplify a polygon using gradient-based filtering
  std::vector<std::pair<double, double>> simplifyPolygon(
      const std::vector<std::pair<double, double>>& polygon, 
      double angleThreshold = 0.05) { // Adjust angle threshold as needed
      if (polygon.size() < 5) return polygon; // Not enough points to simplify

      std::vector<std::pair<double, double>> simplifiedPolygon;
      simplifiedPolygon.push_back(polygon.front()); // Keep first point

      for (size_t i = 1; i < polygon.size() - 1; ++i) {
          double angle = computeAngle(polygon[i - 1], polygon[i], polygon[i + 1]);
          if (angle > angleThreshold) { // Only keep points with significant angle change
              simplifiedPolygon.push_back(polygon[i]);
          }
      }

      simplifiedPolygon.push_back(polygon.back()); // Keep last point

      // Find the two adjacent points that are farthest apart
      size_t farthestIdx = findFarthestAdjacentPoints(simplifiedPolygon);

      // Reorder the polygon so the farthest adjacent points are first and last
      std::vector<std::pair<double, double>> reorderedPolygon;
      reorderedPolygon.insert(reorderedPolygon.begin(), simplifiedPolygon.begin() + farthestIdx + 1, simplifiedPolygon.end());
      reorderedPolygon.insert(reorderedPolygon.end(), simplifiedPolygon.begin(), simplifiedPolygon.begin() + farthestIdx);

      // Ensure the polygon remains **open** by removing duplicate endpoints
      if (reorderedPolygon.front() == reorderedPolygon.back()) {
        reorderedPolygon.pop_back(); // Remove last point if it duplicates the first
      }

      return reorderedPolygon;
  }

  GDS::Structure<NumericType> *getStructure(const std::string &strName) {
    for (size_t i = 0; i < structures.size(); i++) {
      if (strName == structures[i].name) {
        return &structures[i];
      }
    }
    return nullptr;
  }

  void checkReferences() {
    for (auto &str : structures) {
      for (auto &sref : str.sRefs) {
        auto refStr = getStructure(sref.strName);
        refStr->isRef = true;
      }
    }
  }

  void preBuildStructures() {
    for (auto &str : structures) {
      if (str.isRef) {
        if (!str.sRefs.empty()) {
          std::cerr << "Referenced structure contains references" << std::endl;
          continue;
        }

        StructureLayers strLayerMapping;

        for (auto layer : str.containsLayers) {
          strLayerMapping.insert(
              {layer, SmartPointer<viennals::Mesh<NumericType>>::New()});
        }

        for (auto &el : str.elements) {
          SmartPointer<viennals::Mesh<NumericType>> mesh;
          if (el.elementType == GDS::ElementType::elBox) {
            mesh = boxToSurfaceMesh(el, 0, 1, 0, 0);
          } else {
            bool retry = false;
            mesh = polygonToSurfaceMesh(el, 0, 1, 0, 0, retry);
            if (retry) {
              pointOrderFlag = !pointOrderFlag;
              mesh = polygonToSurfaceMesh(el, 0, 1, 0, 0, retry);
            }
          }
          strLayerMapping[el.layer]->append(*mesh);
        }

        assembledStructures.insert({str.name, strLayerMapping});
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
    bounds_[0] = minBounds[0] - boundaryPadding[0];
    bounds_[1] = maxBounds[0] + boundaryPadding[0];
    bounds_[2] = minBounds[1] - boundaryPadding[1];
    bounds_[3] = maxBounds[1] + boundaryPadding[1];
    bounds_[4] = -1.;
    bounds_[5] = 1.;
  }

  void addBox(lsDomainType levelSet, GDS::Element<NumericType> &element,
              const NumericType baseHeight, const NumericType height,
              const NumericType xOffset, const NumericType yOffset) const {
    auto tmpLS = lsDomainType::New(levelSet->getGrid());

    auto minPoint = element.pointCloud[1];
    minPoint[2] = baseHeight;
    auto maxPoint = element.pointCloud[3];
    maxPoint[2] = baseHeight + height;

    viennals::MakeGeometry<NumericType, D>(
        tmpLS, SmartPointer<viennals::Box<NumericType, D>>::New(
                   minPoint.data(), maxPoint.data()))
        .apply();
    viennals::BooleanOperation<NumericType, D>(
        levelSet, tmpLS, viennals::BooleanOperationEnum::UNION)
        .apply();
  }

  void addPolygon(lsDomainType levelSet, GDS::Element<NumericType> &element,
                  const NumericType baseHeight, const NumericType height,
                  const NumericType xOffset, const NumericType yOffset) {
    bool retry = false;
    auto mesh = polygonToSurfaceMesh(element, baseHeight, height, xOffset,
                                     yOffset, retry);
    if (retry) {
      pointOrderFlag = !pointOrderFlag;
      mesh = polygonToSurfaceMesh(element, baseHeight, height, xOffset, yOffset,
                                  retry);
    }
    auto tmpLS = lsDomainType::New(levelSet->getGrid());
    viennals::FromSurfaceMesh<NumericType, D>(tmpLS, mesh).apply();
    viennals::BooleanOperation<NumericType, D>(
        levelSet, tmpLS, viennals::BooleanOperationEnum::UNION)
        .apply();
  }

  SmartPointer<viennals::Mesh<NumericType>>
  boxToSurfaceMesh(GDS::Element<NumericType> &element,
                   const NumericType baseHeight, const NumericType height,
                   const NumericType xOffset, const NumericType yOffset) {
    auto mesh = SmartPointer<viennals::Mesh<NumericType>>::New();

    for (auto &point : element.pointCloud) {
      point[0] += xOffset;
      point[1] += yOffset;
      point[2] = baseHeight;
      mesh->insertNextNode(point);
    }

    for (auto &point : element.pointCloud) {
      point[0] += xOffset;
      point[1] += yOffset;
      point[2] = baseHeight + height;
      mesh->insertNextNode(point);
    }

    for (unsigned i = 0; i < 4; i++) {
      mesh->insertNextTriangle(std::array<unsigned, 3>{i, i + 4, (i + 1) % 4});
      mesh->insertNextTriangle(
          std::array<unsigned, 3>{i + 4, (i + 4 + 1) % 4 + 4, (i + 4 + 1) % 4});
    }
    mesh->insertNextTriangle(std::array<unsigned, 3>{0, 1, 3});
    mesh->insertNextTriangle(std::array<unsigned, 3>{1, 2, 3});
    mesh->insertNextTriangle(std::array<unsigned, 3>{4, 7, 5});
    mesh->insertNextTriangle(std::array<unsigned, 3>{5, 7, 6});

    return mesh;
  }

  SmartPointer<viennals::Mesh<NumericType>>
  polygonToSurfaceMesh(GDS::Element<NumericType> &element,
                       const NumericType baseHeight, const NumericType height,
                       const NumericType xOffset, const NumericType yOffset,
                       bool &retry) {
    auto mesh = SmartPointer<viennals::Mesh<NumericType>>::New();

    unsigned numPointsFlat = element.pointCloud.size();

    // sidewalls
    for (unsigned i = 0; i < numPointsFlat; i++) {
      std::array<NumericType, 3> offsetPoint = element.pointCloud[i];
      offsetPoint[0] += xOffset;
      offsetPoint[1] += yOffset;
      offsetPoint[2] = baseHeight;
      mesh->insertNextNode(offsetPoint);

      if (pointOrderFlag) {
        mesh->insertNextTriangle(std::array<unsigned, 3>{
            i, (i + 1) % numPointsFlat, i + numPointsFlat});
      } else {
        mesh->insertNextTriangle(std::array<unsigned, 3>{
            i + numPointsFlat, (i + 1) % numPointsFlat, i});
      }
    }

    for (unsigned i = 0; i < numPointsFlat; i++) {
      unsigned upPoint = i + numPointsFlat;
      std::array<NumericType, 3> offsetPoint = element.pointCloud[i];
      offsetPoint[0] += xOffset;
      offsetPoint[1] += yOffset;
      offsetPoint[2] = baseHeight + height;
      mesh->insertNextNode(offsetPoint);

      if (pointOrderFlag) {
        mesh->insertNextTriangle(std::array<unsigned, 3>{
            upPoint, (upPoint + 1) % numPointsFlat,
            (upPoint + 1) % numPointsFlat + numPointsFlat});
      } else {
        mesh->insertNextTriangle(std::array<unsigned, 3>{
            (upPoint + 1) % numPointsFlat + numPointsFlat,
            (upPoint + 1) % numPointsFlat, upPoint});
      }
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
    unsigned counter = 0;
    while (numTriangles < (numPointsFlat - 2)) {
      i = rightNeighbors[i];
      if (isEar(leftNeighbors[i], i, rightNeighbors[i], mesh, numPointsFlat)) {
        if (pointOrderFlag) {

          mesh->insertNextTriangle(
              std::array<unsigned, 3>{rightNeighbors[i], i, leftNeighbors[i]});
        } else {
          mesh->insertNextTriangle(
              std::array<unsigned, 3>{leftNeighbors[i], i, rightNeighbors[i]});
        }

        // remove point
        leftNeighbors[rightNeighbors[i]] = leftNeighbors[i];
        rightNeighbors[leftNeighbors[i]] = rightNeighbors[i];

        numTriangles++;
      }

      if (counter++ > triangulationTimeOut) {
        if (!retry) {
          retry = true;
          return mesh;
        } else {
          Logger::getInstance()
              .addError("Timeout in surface triangulation.")
              .print();
        }
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
      std::swap(triangle[0], triangle[2]); // swap for correct orientation
      mesh->insertNextTriangle(triangle);
    }

    return mesh;
  }

  bool isEar(int i, int j, int k,
             SmartPointer<viennals::Mesh<NumericType>> mesh,
             unsigned numPoints) const {
    auto &points = mesh->getNodes();

    // check if triangle is clockwise orientated
    if (((points[i][0] * points[j][1] + points[i][1] * points[k][0] +
          points[j][0] * points[k][1] - points[k][0] * points[j][1] -
          points[k][1] * points[i][0] - points[j][0] * points[i][1]) < 0.) !=
        !pointOrderFlag)
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
        if (((side_1 < eps) && (side_2 < eps) && (side_3 < eps)) ||
            ((side_1 > -eps) && (side_2 > -eps) && (side_3 > -eps)))
          return false;
      }
    }

    return true;
  }

  void adjustPreBuiltMeshHeight(SmartPointer<viennals::Mesh<NumericType>> mesh,
                                const NumericType baseHeight,
                                const NumericType height) const {
    auto &nodes = mesh->getNodes();

    for (auto &n : nodes) {
      if (n[2] < 1.) {
        n[2] = baseHeight;
      } else {
        n[2] = baseHeight + height;
      }
    }
  }

  void resetPreBuiltMeshHeight(SmartPointer<viennals::Mesh<NumericType>> mesh,
                               const NumericType baseHeight,
                               const NumericType height) const {
    auto &nodes = mesh->getNodes();

    for (auto &n : nodes) {
      if (n[2] == baseHeight) {
        n[2] = 0.;
      } else {
        n[2] = 1.;
      }
    }
  }

  static inline NumericType deg2rad(const NumericType angleDeg) {
    return angleDeg * M_PI / 180.;
  }

private:
  std::vector<GDS::Structure<NumericType>> structures;
  std::unordered_map<std::string, StructureLayers> assembledStructures;
  std::array<NumericType, 2> boundaryPadding = {0., 0.};
  std::array<NumericType, 2> minBounds;
  std::array<NumericType, 2> maxBounds;
  static bool pointOrderFlag;
  unsigned triangulationTimeOut = 1000000;
  static constexpr double eps = 1e-6;

  double bounds_[6] = {};
  NumericType gridDelta_ = 1.;
  BoundaryType boundaryConds_[3] = {BoundaryType::REFLECTIVE_BOUNDARY,
                                    BoundaryType::REFLECTIVE_BOUNDARY,
                                    BoundaryType::INFINITE_BOUNDARY};
};

template <class NumericType, int D>
bool GDSGeometry<NumericType, D>::pointOrderFlag = true;

} // namespace viennaps