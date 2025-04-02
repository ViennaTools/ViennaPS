#pragma once

#include "psGDSUtils.hpp"
#include "psGDSMaskProximity.hpp"

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
        // add single layer
        if (auto contains = str.containsLayers.find(layer);
            contains != str.containsLayers.end()) {
          for (auto &el : str.elements) {
            if (el.layer == layer) {
              if (el.elementType == GDS::ElementType::elBox)
                addBox(blurring ? blurredLS : unblurredLS, el, 0., 0.);
              else
                addPolygon(blurring ? blurredLS : unblurredLS, el, 0., 0.);
            }
          }

        }

        for (auto &sref : str.sRefs) {
          auto refStr = getStructure(sref.strName);
          if (!refStr) continue;

          for (auto &el : refStr->elements) {
            if (el.layer != layer)
              continue;
            
            // Apply transformations to element copy
            auto elCopy = el;

            // Magnification
            if (sref.magnification != 0.) {
              for (auto &pt : elCopy.pointCloud) {
                pt[0] *= sref.magnification;
                pt[1] *= sref.magnification;
              }
            }

            // Rotation
            if (sref.angle != 0) {
              double rad = deg2rad(sref.angle);
              double cosA = std::cos(rad);
              double sinA = std::sin(rad);
              for (auto &pt : elCopy.pointCloud) {
                double x = pt[0];
                double y = pt[1];
                pt[0] = x * cosA - y * sinA;
                pt[1] = x * sinA + y * cosA;
              }
            }

            // Translation
            for (auto &pt : elCopy.pointCloud) {
              pt[0] += sref.refPoint[0];
              pt[1] += sref.refPoint[1];
            }

            // Flip along x
            if (sref.flipped) {
              for (auto &pt : elCopy.pointCloud)
                pt[0] = -pt[0];
              // keep winding order
              std::reverse(elCopy.pointCloud.begin(), elCopy.pointCloud.end());
            }

            // Add element to layer
            if (el.elementType == GDS::ElementType::elBox)
              addBox(blurring ? blurredLS : unblurredLS, elCopy, 0., 0.);
            else
              addPolygon(blurring ? blurredLS : unblurredLS, elCopy, 0., 0.);
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
    calculateBoundingBoxes();
  }

  std::set<int16_t> getAllLayers() const {
    std::set<int16_t> allLayers;
    for (const auto &str : structures) {
      allLayers.insert(str.containsLayers.begin(), str.containsLayers.end());
    }
    return allLayers;
  }

  std::size_t getNumberOfStructures() const {
    return structures.size();
  }

private:

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
        if (refStr)
          refStr->isRef = true;
        else
          Logger::getInstance().addWarning("Missing referenced structure: " + sref.strName).print();
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

  static inline NumericType deg2rad(const NumericType angleDeg) {
    return angleDeg * M_PI / 180.;
  }

private:
  std::vector<GDS::Structure<NumericType>> structures;
  std::unordered_map<std::string, StructureLayers> assembledStructures;
  std::array<NumericType, 2> boundaryPadding = {0., 0.};
  std::array<NumericType, 2> minBounds;
  std::array<NumericType, 2> maxBounds;

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

};

} // namespace viennaps