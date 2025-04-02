#pragma once

#include "psGDSUtils.hpp"
#include "psGDSMaskProximity.hpp"
// #include "psLaplacianSmoothing.hpp"

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsGeometries.hpp>
#include <lsMakeGeometry.hpp>
#include <lsTransformMesh.hpp>

#include <lsVTKWriter.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsToMesh.hpp>

#include <lsExtrude.hpp>

#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>

namespace ls = viennals;

namespace viennaps {
  
using namespace viennacore;

template <class NumericType, int D = 3> class GDSGeometry {
  using StructureLayers =
      std::unordered_map<int16_t, SmartPointer<ls::Mesh<NumericType>>>;
  using lsDomainType = SmartPointer<ls::Domain<NumericType, D>>;
  using lsDomainType2D = SmartPointer<ls::Domain<NumericType, 2>>;
  using BoundaryType = typename ls::Domain<NumericType, D>::BoundaryType;
  using PointType = typename std::vector<std::pair<hrleVectorType<hrleIndexType, 2>, NumericType>>;

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

  NumericType getGridDelta() const { return gridDelta_; }

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
                               const NumericType height, 
                               bool mask = false, bool blurring = true) {

    blurring = blurring && blur;
    auto levelSet = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
    lsDomainType2D unblurredLS = lsDomainType2D::New(bounds_, boundaryConds_, gridDelta_);
    lsDomainType2D blurredLS = lsDomainType2D::New(bounds_, boundaryConds_, exposureDelta);
    
    for (auto &str : structures) { // loop over all structures
      if (!str.isRef) {
        // add single elements
        if (auto contains = str.containsLayers.find(layer);
            contains != str.containsLayers.end()) {
          for (auto &el : str.elements) {
            if (el.layer == layer) {
              if (el.elementType == GDS::ElementType::elBox) {
                if (blurring)
                  addBox(blurredLS, el, 0., 0.);
                else
                  addBox(unblurredLS, el, 0., 0.);
              } else {
                if (blurring)
                  addPolygon(blurredLS, el, 0, 0);
                else
                  addPolygon(unblurredLS, el, 0, 0);
              }
            }
          }

          std::array<NumericType, 2> extrudeExtent = {baseHeight, baseHeight + height};

          if (blurring) {
            PointType pointData =  applyBlur(blurredLS);
            blurredLS->insertPoints(pointData);
            blurredLS->finalize(2);
            ls::Expand<double, 2>(blurredLS, 2).apply();

            ls::Extrude<NumericType>(blurredLS, levelSet, extrudeExtent).apply();
          } else {
            ls::Extrude<NumericType>(unblurredLS, levelSet, extrudeExtent).apply();
          }
        }

        // add structure references
        auto strMesh = SmartPointer<ls::Mesh<NumericType>>::New();
        for (auto &sref : str.sRefs) {
          auto refStr = getStructure(sref.strName);
          if (auto contains = refStr->containsLayers.find(layer);
              contains != refStr->containsLayers.end()) {
            assert(assembledStructures[refStr->name][layer]);

            // copy mesh here
            auto copy = assembledStructures[refStr->name][layer];
            auto preBuiltStrMesh =
                SmartPointer<ls::Mesh<NumericType>>::New();
            preBuiltStrMesh->nodes = copy->nodes;
            preBuiltStrMesh->triangles = copy->triangles;
            adjustPreBuiltMeshHeight(preBuiltStrMesh, baseHeight, height);

            if (sref.angle > 0.) {
              ls::TransformMesh<NumericType>(
                  preBuiltStrMesh, ls::TransformEnum::ROTATION,
                  hrleVectorType<double, 3>{0., 0., 1.}, deg2rad(sref.angle))
                  .apply();
            }

            if (sref.magnification > 0.) {
              ls::TransformMesh<NumericType>(
                  preBuiltStrMesh, ls::TransformEnum::SCALE,
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

            ls::TransformMesh<NumericType>(
                preBuiltStrMesh, ls::TransformEnum::TRANSLATION,
                hrleVectorType<double, 3>{sref.refPoint[0], sref.refPoint[1],
                                          0.})
                .apply();

            strMesh->append(*preBuiltStrMesh);
          }
        }
        if (strMesh->nodes.size() > 0) {
          auto tmpLS = lsDomainType::New(levelSet->getGrid());
          ls::FromSurfaceMesh<NumericType, D>(tmpLS, strMesh).apply();
          ls::BooleanOperation<NumericType, D>(
              levelSet, tmpLS, ls::BooleanOperationEnum::UNION)
              .apply();
        }
      } // if (!str.isRef)
    } // for (auto &str : structures)
    

    if (mask) {
      // Create bottom substrate (z >= 0)
      auto bottomLS = lsDomainType::New(levelSet->getGrid());
      double originLow[3] = {0., 0., baseHeight};
      double normalLow[3] = {0., 0., -1.};
      auto bottomPlane = ls::SmartPointer<ls::Plane<double, D>>::New(originLow, normalLow);
      ls::MakeGeometry<double, 3>(bottomLS, bottomPlane).apply();

      // Create top cap (z <= 5 µm)
      auto topLS = lsDomainType::New(levelSet->getGrid());
      double originHigh[3] = {0., 0., baseHeight + height}; // Adjust to match extrusion
      double normalHigh[3] = {0., 0., 1.};
      auto topPlane = ls::SmartPointer<ls::Plane<double, D>>::New(originHigh, normalHigh);
      ls::MakeGeometry<double, D>(topLS, topPlane).apply();

      // Intersect with bottom
      ls::BooleanOperation<double, D>(levelSet, bottomLS, ls::BooleanOperationEnum::INTERSECT).apply();
      // Intersect with top
      ls::BooleanOperation<double, D>(levelSet, topLS, ls::BooleanOperationEnum::INTERSECT).apply();
    } // if(mask) end
    return levelSet;
  } // function end

  PointType applyBlur(lsDomainType2D blurredLS) {
    GDSMaskProximity<NumericType> proximity(blurredLS, exposureDelta, sigmas, weights);
    proximity.apply();

    auto exposureGrid = proximity.getExposureGrid();
    if (lsInternal::Logger::getLogLevel() >= 2)
      proximity.saveGridToCSV(exposureGrid, "finalGrid.csv");

    int gridSizeY = exposureGrid.size();
    int gridSizeX = exposureGrid[0].size();

    PointType pointData;
    std::vector<std::pair<int, int>> directions = {
      {-1, 0}, {1, 0}, {0, -1}, {0, 1}  // 4-neighbor stencil
    };
    
    for (int y = 0; y < gridSizeY; ++y) {
      for (int x = 0; x < gridSizeX; ++x) {
        double current = exposureGrid[y][x];

        // Check if point (x,y) is on the contour
        if (current == threshold) {
          double xReal = x * exposureDelta + bounds_[0];
          int xLS = std::round(xReal / gridDelta_);
          double yReal = y * exposureDelta + bounds_[2];
          int yLS = std::round(yReal / gridDelta_);
          hrleVectorType<hrleIndexType, 3> pos;
          pos[0] = xLS; pos[1] = yLS;
          pointData.emplace_back(pos, 0.);
          break;
        }
    
        double minDist = std::numeric_limits<double>::max();
        int bestNx = -1, bestNy = -1;
        
        for (auto [dy, dx] : directions) {
          int ny = y + dy;
          int nx = x + dx;

          if ((nx < 0 || nx >= gridSizeX || ny < 0 || ny >= gridSizeY) &&
              !applyBoundaryCondition(nx, ny, gridSizeX, gridSizeY, boundaryConds_))
            continue;

          double neighbor = exposureGrid[ny][nx];

          // Check if neighbor (nx,ny) is on the contour
          if (neighbor == threshold) {
            double xReal = nx * exposureDelta + bounds_[0];
            int xLS = std::round(xReal / gridDelta_);
            double yReal = ny * exposureDelta + bounds_[2];
            int yLS = std::round(yReal / gridDelta_);
            hrleVectorType<hrleIndexType, 3> pos;
            pos[0] = xLS; pos[1] = yLS;
            pointData.emplace_back(pos, 0.);
            break;
          }
           
          // Check if neighbors are on opposite sites of contour
          if ((current - threshold) * (neighbor - threshold) < 0) {
            // Interpolate sub-cell distance
            double dist = std::abs((threshold - current) / (neighbor - current));

            if (dist < minDist) {
              minDist = dist;
              bestNx = nx;
              bestNy = ny;
            }
          }
        }

        if ((minDist < 1.0) && (bestNx >= 0.) && (bestNy >= 0.)) {
          double sdfCurrent  = minDist * exposureDelta;

          double xReal = x * exposureDelta + bounds_[0];
          double yReal = y * exposureDelta + bounds_[2];

          int xIndex = std::round(xReal / gridDelta_);
          int yIndex = std::round(yReal / gridDelta_);

          hrleVectorType<hrleIndexType, 3> curIndex;
          curIndex[0] = xIndex; curIndex[1] = yIndex;

          double sign = (current < threshold) ? 1.0 : -1.0;
          pointData.emplace_back(curIndex, sign * sdfCurrent);
        }
      } // end x iter
    } // end y iter
    return pointData;
  }

  void addBlur(std::vector<NumericType> inSigmas,
               std::vector<NumericType> inWeights,
               NumericType inThreshold = 0.5,
               NumericType delta = 0.) {
    sigmas = inSigmas;
    weights = inWeights;
    threshold = inThreshold;
    exposureDelta = delta;
    blur = true;
  }

  void printBound() const {
    std::cout << "Geometry: (" << minBounds[0] << ", " << minBounds[1]
              << ") - (" << maxBounds[0] << ", " << maxBounds[1] << ")"
              << std::endl;
  }

  std::array<std::array<NumericType, 2>, 2> getBoundingBox() const {
    return {minBounds, maxBounds};
  }

  auto getBounds() const { return bounds_; }

  void insertNextStructure(GDS::Structure<NumericType> const &structure) {
    structures.push_back(structure);
  }

  void finalize() {
    checkReferences();
    // preBuildStructures();
    calculateBoundingBoxes();
  }

private:

  void applyBlur() {

  }

  bool applyBoundaryCondition(
      int &x, int &y,
      int maxX, int maxY,
      const BoundaryType boundaryConditions[]) {
    // X
    if (x < 0) {
      if (boundaryConditions[0] == BoundaryType::INFINITE_BOUNDARY)
        return false;
      else if (boundaryConditions[0] == BoundaryType::REFLECTIVE_BOUNDARY)
        x = -x;
      else if (boundaryConditions[0] == BoundaryType::PERIODIC_BOUNDARY)
        x = maxX - 1;
    } else if (x >= maxX) {
      if (boundaryConditions[0] == BoundaryType::INFINITE_BOUNDARY)
        return false;
      else if (boundaryConditions[0] == BoundaryType::REFLECTIVE_BOUNDARY)
        x = 2 * maxX - x - 2;
      else if (boundaryConditions[0] == BoundaryType::PERIODIC_BOUNDARY)
        x = 0;
    }

    // Y
    if (y < 0) {
      if (boundaryConditions[1] == BoundaryType::INFINITE_BOUNDARY)
        return false;
      else if (boundaryConditions[1] == BoundaryType::REFLECTIVE_BOUNDARY)
        y = -y;
      else if (boundaryConditions[1] == BoundaryType::PERIODIC_BOUNDARY)
        y = maxY - 1;
    } else if (y >= maxY) {
      if (boundaryConditions[1] == BoundaryType::INFINITE_BOUNDARY)
        return false;
      else if (boundaryConditions[1] == BoundaryType::REFLECTIVE_BOUNDARY)
        y = 2 * maxY - y - 2;
      else if (boundaryConditions[1] == BoundaryType::PERIODIC_BOUNDARY)
        y = 0;
    }

    return true;
  }


  // // Function to compute angle change between three points
  // double computeAngle(std::pair<double, double> p1, 
  //                     std::pair<double, double> p2, 
  //                     std::pair<double, double> p3) {
  //     double v1x = p2.first - p1.first;
  //     double v1y = p2.second - p1.second;
  //     double v2x = p3.first - p2.first;
  //     double v2y = p3.second - p2.second;

  //     double dotProduct = v1x * v2x + v1y * v2y;
  //     double magnitude1 = std::sqrt(v1x * v1x + v1y * v1y);
  //     double magnitude2 = std::sqrt(v2x * v2x + v2y * v2y);

  //     if (magnitude1 == 0.0 || magnitude2 == 0.0) return 0.0; // Avoid division by zero

  //     double cosTheta = dotProduct / (magnitude1 * magnitude2);
  //     return std::acos(std::clamp(cosTheta, -1.0, 1.0)); // Clamp to avoid NaN
  // }

  // // Function to compute Euclidean distance
  // double euclideanDistance(const std::pair<double, double>& p1, const std::pair<double, double>& p2) {
  //   return std::sqrt(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2));
  // }

  // // Function to find the two adjacent points in the polygon that are farthest apart
  // size_t findFarthestAdjacentPoints(const std::vector<std::pair<double, double>>& polygon) {
  //   if (polygon.size() < 2) return 0; // Not enough points to process

  //   size_t maxIdx = 0;
  //   double maxDistance = 0.0;

  //   for (size_t i = 0; i < polygon.size() - 1; ++i) {
  //       double dist = euclideanDistance(polygon[i], polygon[i + 1]);
  //       if (dist > maxDistance) {
  //           maxDistance = dist;
  //           maxIdx = i;
  //       }
  //   }
  //   return maxIdx;  // Return index of the first point in the farthest pair
  // }

  // // Function to simplify a polygon using gradient-based filtering
  // std::vector<std::pair<double, double>> simplifyPolygon(
  //   const std::vector<std::pair<double, double>>& polygon, 
  //   double angleThreshold = 0.05, 
  //   double distanceThreshold = 0.5) {  // Adjust angle threshold and grid spacing as needed

  //   if (polygon.size() < 5) return polygon; // Not enough points to simplify

  //   std::vector<std::pair<double, double>> simplifiedPolygon;
  //   simplifiedPolygon.push_back(polygon.front()); // Keep first point

  //   for (size_t i = 1; i < polygon.size() - 1; ++i) {
  //       double angle = computeAngle(polygon[i - 1], polygon[i], polygon[i + 1]);

  //     if (angle > angleThreshold && euclideanDistance(polygon[i], simplifiedPolygon.back()) >= distanceThreshold) {          
  //           simplifiedPolygon.push_back(polygon[i]);
  //       }
  //   }

  //   simplifiedPolygon.push_back(polygon.back()); // Keep last point

  //   // Find the two adjacent points that are farthest apart
  //   size_t farthestIdx = findFarthestAdjacentPoints(simplifiedPolygon);

  //   // Reorder the polygon so the farthest adjacent points are first and last
  //   std::vector<std::pair<double, double>> reorderedPolygon;
  //   reorderedPolygon.insert(reorderedPolygon.begin(), simplifiedPolygon.begin() + farthestIdx + 1, simplifiedPolygon.end());
  //   reorderedPolygon.insert(reorderedPolygon.end(), simplifiedPolygon.begin(), simplifiedPolygon.begin() + farthestIdx);

  //   // Ensure the polygon remains open by removing duplicate endpoints
  //   if (reorderedPolygon.front() == reorderedPolygon.back()) {
  //     reorderedPolygon.pop_back();
  //   }

  //   return reorderedPolygon;
  // }

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

  // void preBuildStructures() {
  //   for (auto &str : structures) {
  //     if (str.isRef) {
  //       if (!str.sRefs.empty()) {
  //         std::cerr << "Referenced structure contains references" << std::endl;
  //         continue;
  //       }

  //       StructureLayers strLayerMapping;

  //       for (auto layer : str.containsLayers) {
  //         strLayerMapping.insert(
  //             {layer, SmartPointer<ls::Mesh<NumericType>>::New()});
  //       }

  //       for (auto &el : str.elements) {
  //         SmartPointer<ls::Mesh<NumericType>> mesh;
  //         if (el.elementType == GDS::ElementType::elBox) {
  //           mesh = boxToSurfaceMesh(el, 0, 1, 0, 0);
  //         } else {
  //           bool retry = false;
  //           // mesh = polygonToSurfaceMesh(el, 0, 1, 0, 0, retry);
  //           mesh = polygonToSurfaceMesh(el, 0, 0);
  //           if (retry) {
  //             pointOrderFlag = !pointOrderFlag;
  //             // mesh = polygonToSurfaceMesh(el, 0, 1, 0, 0, retry);
  //             mesh = polygonToSurfaceMesh(el, 0, 0);
  //           }
  //         }
  //         strLayerMapping[el.layer]->append(*mesh);
  //       }

  //       assembledStructures.insert({str.name, strLayerMapping});
  //     }
  //   }
  // }

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

  void addBox(lsDomainType2D layer2D, GDS::Element<NumericType> &element,
              const NumericType xOffset, const NumericType yOffset) const {

    assert(element.elementType == GDS::ElementType::elBox);
    assert(element.pointCloud.size() == 4); // GDSII box is a rectangle

    using VectorType = hrleVectorType<NumericType, 2>;

    // The corners in GDS are typically ordered clockwise or counter-clockwise
    VectorType minCorner{
        std::min({element.pointCloud[0][0], element.pointCloud[1][0],
                  element.pointCloud[2][0], element.pointCloud[3][0]}),
        std::min({element.pointCloud[0][1], element.pointCloud[1][1],
                  element.pointCloud[2][1], element.pointCloud[3][1]})
    };

    VectorType maxCorner{
        std::max({element.pointCloud[0][0], element.pointCloud[1][0],
                  element.pointCloud[2][0], element.pointCloud[3][0]}),
        std::max({element.pointCloud[0][1], element.pointCloud[1][1],
                  element.pointCloud[2][1], element.pointCloud[3][1]})
    };

    // Generate a level set box using MakeGeometry
    ls::MakeGeometry<NumericType, 2>(
        layer2D,
        SmartPointer<ls::Box<NumericType, 2>>::New(minCorner, maxCorner))
        .apply();
  }
  
  // void addBox(lsDomainType levelSet, GDS::Element<NumericType> &element,
  //             const NumericType baseHeight, const NumericType height,
  //             const NumericType xOffset, const NumericType yOffset) const {
  //   auto tmpLS = lsDomainType::New(levelSet->getGrid());

  //   auto minPoint = element.pointCloud[1];
  //   minPoint[2] = baseHeight;
  //   auto maxPoint = element.pointCloud[3];
  //   maxPoint[2] = baseHeight + height;

  //   ls::MakeGeometry<NumericType, D>(
  //       tmpLS, SmartPointer<ls::Box<NumericType, D>>::New(
  //                  minPoint.data(), maxPoint.data()))
  //       .apply();
  //   ls::BooleanOperation<NumericType, 2>(
  //       levelSet, tmpLS, ls::BooleanOperationEnum::UNION)
  //       .apply();
  // }

  void addPolygon(lsDomainType2D layer2D,
                  const GDS::Element<NumericType> &element,
                  const NumericType xOffset,
                  const NumericType yOffset) {

    // Create a 2D level set from the polygon
    auto mesh = polygonToSurfaceMesh(element, xOffset, yOffset);
    lsDomainType2D tmpLS = lsDomainType2D::New(getBounds(), boundaryConds_, getGridDelta());
    ls::FromSurfaceMesh<NumericType, 2>(tmpLS, mesh).apply();

    ls::BooleanOperation<NumericType, 2>(
      layer2D, tmpLS, ls::BooleanOperationEnum::UNION)
      .apply();
  }

  // void addPolygon(lsDomainType levelSet,
  //                 const GDS::Element<NumericType> &element,
  //                 const NumericType baseHeight,
  //                 const NumericType height,
  //                 const NumericType xOffset,
  //                 const NumericType yOffset) {

  //   // Create a 2D level set from the polygon
  //   auto mesh = polygonToSurfaceMesh(element, xOffset, yOffset);
  //   lsDomainType2D tmpLS = lsDomainType2D::New(getBounds(), boundaryConds_, getGridDelta());
  //   ls::FromSurfaceMesh<NumericType, 2>(tmpLS, mesh).apply();

  //   auto levelSet3D = lsDomainType::New(levelSet->getGrid());
  //   std::array<NumericType, 2> extrudeExtent = {baseHeight, baseHeight + height};
  //   ls::Extrude<NumericType>(tmpLS, levelSet3D, extrudeExtent).apply();
   
  //   ls::BooleanOperation<NumericType, D>(
  //       levelSet, levelSet3D, ls::BooleanOperationEnum::UNION)
  //       .apply();
  // }

  // SmartPointer<ls::Mesh<NumericType>>
  // boxToSurfaceMesh(GDS::Element<NumericType> &element,
  //                  const NumericType baseHeight, const NumericType height,
  //                  const NumericType xOffset, const NumericType yOffset) {
  //   auto mesh = SmartPointer<ls::Mesh<NumericType>>::New();

  //   for (auto &point : element.pointCloud) {
  //     point[0] += xOffset;
  //     point[1] += yOffset;
  //     point[2] = baseHeight;
  //     mesh->insertNextNode(point);
  //   }

  //   for (auto &point : element.pointCloud) {
  //     point[0] += xOffset;
  //     point[1] += yOffset;
  //     point[2] = baseHeight + height;
  //     mesh->insertNextNode(point);
  //   }

  //   for (unsigned i = 0; i < 4; i++) {
  //     mesh->insertNextTriangle(std::array<unsigned, 3>{i, i + 4, (i + 1) % 4});
  //     mesh->insertNextTriangle(
  //         std::array<unsigned, 3>{i + 4, (i + 4 + 1) % 4 + 4, (i + 4 + 1) % 4});
  //   }
  //   mesh->insertNextTriangle(std::array<unsigned, 3>{0, 1, 3});
  //   mesh->insertNextTriangle(std::array<unsigned, 3>{1, 2, 3});
  //   mesh->insertNextTriangle(std::array<unsigned, 3>{4, 7, 5});
  //   mesh->insertNextTriangle(std::array<unsigned, 3>{5, 7, 6});

  //   return mesh;
  // }

  SmartPointer<ls::Mesh<NumericType>>
  polygonToSurfaceMesh(const GDS::Element<NumericType> &element,
                       const NumericType xOffset, const NumericType yOffset) {
    auto mesh = SmartPointer<ls::Mesh<NumericType>>::New();
    const auto &points = element.pointCloud;
  
    if (points.size() < 2) {
      return mesh;
    }
  
    // --- Determine winding order (signed area) ---
    double signedArea = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
      const auto &p1 = points[i];
      const auto &p2 = points[(i + 1) % points.size()];
      signedArea += (p1[0] * p2[1] - p2[0] * p1[1]);
    }
    bool isCCW = (signedArea > 0);
  
    // --- Add points to mesh with offsets ---
    std::vector<unsigned> indices;
    for (const auto &pt : points) {
      indices.push_back(mesh->insertNextNode(pt));
    }

    // --- Add lines ---
    for (size_t i = 1; i < indices.size(); ++i) {
      if (isCCW)
        mesh->insertNextLine({indices[i], indices[i - 1]});
      else
        mesh->insertNextLine({indices[i - 1], indices[i]});
    }
  
    // --- Close the polygon ---
    if (points.front() != points.back()) {
      if (isCCW)
        mesh->insertNextLine({indices.front(), indices.back()});
      else
        mesh->insertNextLine({indices.back(), indices.front()});
    }
      return mesh;
  }

  // SmartPointer<ls::Mesh<NumericType>>
  // polygonToSurfaceMeshGeom(GDS::Element<NumericType> &element,
  //                      const NumericType baseHeight, const NumericType height,
  //                      const NumericType xOffset, const NumericType yOffset,
  //                      bool &retry) {
  //   auto mesh = SmartPointer<ls::Mesh<NumericType>>::New();

  //   unsigned numPointsFlat = element.pointCloud.size();

  //   // sidewalls
  //   for (unsigned i = 0; i < numPointsFlat; i++) {
  //     std::array<NumericType, 3> offsetPoint = element.pointCloud[i];
  //     offsetPoint[0] += xOffset;
  //     offsetPoint[1] += yOffset;
  //     offsetPoint[2] = baseHeight;
  //     mesh->insertNextNode(offsetPoint);

  //     if (pointOrderFlag) {
  //       mesh->insertNextTriangle(std::array<unsigned, 3>{
  //           i, (i + 1) % numPointsFlat, i + numPointsFlat});
  //     } else {
  //       mesh->insertNextTriangle(std::array<unsigned, 3>{
  //           i + numPointsFlat, (i + 1) % numPointsFlat, i});
  //     }
  //   }

  //   for (unsigned i = 0; i < numPointsFlat; i++) {
  //     unsigned upPoint = i + numPointsFlat;
  //     std::array<NumericType, 3> offsetPoint = element.pointCloud[i];
  //     offsetPoint[0] += xOffset;
  //     offsetPoint[1] += yOffset;
  //     offsetPoint[2] = baseHeight + height;
  //     mesh->insertNextNode(offsetPoint);

  //     if (pointOrderFlag) {
  //       mesh->insertNextTriangle(std::array<unsigned, 3>{
  //           upPoint, (upPoint + 1) % numPointsFlat,
  //           (upPoint + 1) % numPointsFlat + numPointsFlat});
  //     } else {
  //       mesh->insertNextTriangle(std::array<unsigned, 3>{
  //           (upPoint + 1) % numPointsFlat + numPointsFlat,
  //           (upPoint + 1) % numPointsFlat, upPoint});
  //     }
  //   }

  //   // polygon triangulation (ear clipping algorithm)
  //   std::vector<unsigned> leftNeighbors(numPointsFlat);
  //   std::vector<unsigned> rightNeighbors(numPointsFlat);

  //   // initialize neighbors
  //   for (unsigned i = 0; i < numPointsFlat; i++) {
  //     leftNeighbors[i] = ((i - 1) + numPointsFlat) % numPointsFlat;
  //     rightNeighbors[i] = ((i + 1) + numPointsFlat) % numPointsFlat;
  //   }

  //   unsigned numTriangles = 0;
  //   unsigned i = numPointsFlat - 1;
  //   unsigned counter = 0;
  //   while (numTriangles < (numPointsFlat - 2)) {
  //     i = rightNeighbors[i];
  //     if (isEar(leftNeighbors[i], i, rightNeighbors[i], mesh, numPointsFlat)) {
  //       if (pointOrderFlag) {

  //         mesh->insertNextTriangle(
  //             std::array<unsigned, 3>{rightNeighbors[i], i, leftNeighbors[i]});
  //       } else {
  //         mesh->insertNextTriangle(
  //             std::array<unsigned, 3>{leftNeighbors[i], i, rightNeighbors[i]});
  //       }

  //       // remove point
  //       leftNeighbors[rightNeighbors[i]] = leftNeighbors[i];
  //       rightNeighbors[leftNeighbors[i]] = rightNeighbors[i];

  //       numTriangles++;
  //     }

  //     if (counter++ > triangulationTimeOut) {
  //       if (!retry) {
  //         retry = true;
  //         return mesh;
  //       } else {
  //         Logger::getInstance()
  //             .addError("Timeout in surface triangulation.")
  //             .print();
  //       }
  //     }
  //   }

  //   // use same triangles for other side
  //   auto &triangles = mesh->template getElements<3>();
  //   unsigned numPrevTriangles = triangles.size() - 1;
  //   for (unsigned j = 0; j < numTriangles; j++) {
  //     auto triangle = triangles[numPrevTriangles - j];
  //     for (int d = 0; d < 3; d++) {
  //       triangle[d] += numPointsFlat;
  //     }
  //     std::swap(triangle[0], triangle[2]); // swap for correct orientation
  //     mesh->insertNextTriangle(triangle);
  //   }

  //   return mesh;
  // }

  // bool isEar(int i, int j, int k,
  //            SmartPointer<ls::Mesh<NumericType>> mesh,
  //            unsigned numPoints) const {
  //   auto &points = mesh->getNodes();

  //   // check if triangle is clockwise orientated
  //   if (((points[i][0] * points[j][1] + points[i][1] * points[k][0] +
  //         points[j][0] * points[k][1] - points[k][0] * points[j][1] -
  //         points[k][1] * points[i][0] - points[j][0] * points[i][1]) < 0.) !=
  //       !pointOrderFlag)
  //     return false;

  //   for (unsigned m = 0; m < numPoints; m++) {
  //     if ((m != i) && (m != j) && (m != k)) {
  //       // check if point in triangle
  //       auto side_1 =
  //           (points[m][0] - points[j][0]) * (points[i][1] - points[j][1]) -
  //           (points[i][0] - points[j][0]) * (points[m][1] - points[j][1]);
  //       // Segment B to C
  //       auto side_2 =
  //           (points[m][0] - points[k][0]) * (points[j][1] - points[k][1]) -
  //           (points[j][0] - points[k][0]) * (points[m][1] - points[k][1]);
  //       // Segment C to A
  //       auto side_3 =
  //           (points[m][0] - points[i][0]) * (points[k][1] - points[i][1]) -
  //           (points[k][0] - points[i][0]) * (points[m][1] - points[i][1]);

  //       // All the signs must be positive or all negative
  //       if (((side_1 < eps) && (side_2 < eps) && (side_3 < eps)) ||
  //           ((side_1 > -eps) && (side_2 > -eps) && (side_3 > -eps)))
  //         return false;
  //     }
  //   }

  //   return true;
  // }

  void adjustPreBuiltMeshHeight(SmartPointer<ls::Mesh<NumericType>> mesh,
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

  // void resetPreBuiltMeshHeight(SmartPointer<ls::Mesh<NumericType>> mesh,
  //                              const NumericType baseHeight,
  //                              const NumericType height) const {
  //   auto &nodes = mesh->getNodes();

  //   for (auto &n : nodes) {
  //     if (n[2] == baseHeight) {
  //       n[2] = 0.;
  //     } else {
  //       n[2] = 1.;
  //     }
  //   }
  // }

  static inline NumericType deg2rad(const NumericType angleDeg) {
    return angleDeg * M_PI / 180.;
  }

private:
  std::vector<GDS::Structure<NumericType>> structures;
  std::unordered_map<std::string, StructureLayers> assembledStructures;
  std::array<NumericType, 2> boundaryPadding = {0., 0.};
  std::array<NumericType, 2> minBounds;
  std::array<NumericType, 2> maxBounds;
  // static bool pointOrderFlag;
  // unsigned triangulationTimeOut = 1000000;
  // static constexpr double eps = 1e-6;

  double bounds_[6] = {};
  NumericType gridDelta_ = 1.;
  BoundaryType boundaryConds_[3] = {BoundaryType::REFLECTIVE_BOUNDARY,
                                    BoundaryType::REFLECTIVE_BOUNDARY,
                                    BoundaryType::INFINITE_BOUNDARY};

  bool blur = false;
  std::vector<NumericType> sigmas;
  std::vector<NumericType> weights;
  NumericType threshold;
  NumericType exposureDelta;

  // template <class NumericType, int D>
  // bool GDSGeometry<NumericType, D>::pointOrderFlag = true;
};

} // namespace viennaps