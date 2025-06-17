#pragma once

#include "psGDSMaskProximity.hpp"
#include "psGDSUtils.hpp"

#include <lsBooleanOperation.hpp>
#include <lsCheck.hpp>
#include <lsDomain.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsGeometries.hpp>
#include <lsMakeGeometry.hpp>
#include <lsTransformMesh.hpp>

#include <lsToMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>

#include <lsExtrude.hpp>

#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>

namespace ls = viennals;

namespace viennaps {

using namespace viennacore;

template <class NumericType, int D> class GDSGeometry {
  using StructureLayers =
      std::unordered_map<int16_t, SmartPointer<viennals::Mesh<NumericType>>>;
  using lsDomainType = SmartPointer<viennals::Domain<NumericType, D>>;
  using lsDomainType2D = SmartPointer<viennals::Domain<NumericType, 2>>;

  using PointType =
      typename viennals::Domain<NumericType, 2>::PointValueVectorType;
  using IndexType = typename viennahrle::Index<2>;

public:
  GDSGeometry() {}

  explicit GDSGeometry(const NumericType gridDelta) : gridDelta_(gridDelta) {}

  explicit GDSGeometry(const NumericType gridDelta,
                       BoundaryType boundaryConds[D])
      : gridDelta_(gridDelta) {
    for (int i = 0; i < D; ++i)
      boundaryConds_[i] = boundaryConds[i];
  }

  void setGridDelta(const NumericType gridDelta) { gridDelta_ = gridDelta; }

  NumericType getGridDelta() const { return gridDelta_; }

  void setBoundaryPadding(const NumericType xPadding,
                          const NumericType yPadding) {
    boundaryPadding[0] = xPadding;
    boundaryPadding[1] = yPadding;
  }

  void setBoundaryConditions(BoundaryType boundaryConds[D]) {
    for (int i = 0; i < D; i++)
      boundaryConds_[i] = boundaryConds[i];
  }

  void print() const {
    std::cout << "======== STRUCTURES ========" << std::endl;
    for (auto &s : structures) {
      s.print();
    }
    std::cout << "============================" << std::endl;
  }

  // 2D version
  template <int Dim = D, typename std::enable_if<Dim == 2, int>::type = 0>
  lsDomainType layerToLevelSet(const int16_t layer, bool blurLayer = true) {
    return getMaskLevelSet(layer, blurLayer);
  }

  // 3D version
  template <int Dim = D, typename std::enable_if<Dim == 3, int>::type = 0>
  lsDomainType layerToLevelSet(const int16_t layer,
                               const NumericType baseHeight = 0.,
                               const NumericType height = 1., bool mask = false,
                               bool blurLayer = true) {

    // Create a 3D level set from the 2D level set and return it
    auto GDSLevelSet = getMaskLevelSet(layer, blurLayer);
    viennals::Check<NumericType, 2>(GDSLevelSet).apply();
    auto levelSet = lsDomainType::New(bounds_, boundaryConds_, gridDelta_);
    viennals::Extrude<NumericType>(
        GDSLevelSet, levelSet, {baseHeight - gridDelta_, height + gridDelta_})
        .apply();

    if (mask) {
      viennals::BooleanOperation<NumericType, D>(
          levelSet, viennals::BooleanOperationEnum::INVERT)
          .apply();
    }

    // Create bottom substrate
    auto bottomLS = lsDomainType::New(levelSet->getGrid());
    double originLow[3] = {0., 0., baseHeight};
    double normalLow[3] = {0., 0., -1.};
    auto bottomPlane =
        viennals::Plane<NumericType, D>::New(originLow, normalLow);
    viennals::MakeGeometry<NumericType, 3>(bottomLS, bottomPlane).apply();

    // Create top cap
    auto topLS = lsDomainType::New(levelSet->getGrid());
    NumericType originHigh[3] = {
        0., 0., baseHeight + height}; // Adjust to match extrusion
    NumericType normalHigh[3] = {0., 0., 1.};
    auto topPlane =
        viennals::Plane<NumericType, D>::New(originHigh, normalHigh);
    viennals::MakeGeometry<NumericType, D>(topLS, topPlane).apply();

    // Intersect with bottom
    viennals::BooleanOperation<NumericType, D>(
        levelSet, bottomLS, viennals::BooleanOperationEnum::INTERSECT)
        .apply();

    // Intersect with top
    viennals::BooleanOperation<NumericType, D>(
        levelSet, topLS, viennals::BooleanOperationEnum::INTERSECT)
        .apply();

    return levelSet;
  }

  PointType applyBlur(lsDomainType2D blurringLS) {
    // Apply Gaussian blur based on the GDS-extracted "blurringLS"
    GDSMaskProximity<NumericType> proximity(blurringLS, gridRefinement, sigmas,
                                            weights);
    proximity.apply();

    // exposureGrid is the final blurred grid to be used for SDF calculation
    // auto exposureGrid = proximity.getExposedGrid();
    if (lsInternal::Logger::getLogLevel() >= 2)
      proximity.saveGridToCSV("finalGrid.csv");

    PointType pointData;
    const std::vector<std::pair<int, int>> directions = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1} // 4-neighbor stencil
    };

    // Calculate grid bounds
    const int xStart = std::floor(minBounds[0] / gridDelta_);
    const int xEnd = std::ceil(maxBounds[0] / gridDelta_);
    const int yStart = std::floor(minBounds[1] / gridDelta_);
    const int yEnd = std::ceil(maxBounds[1] / gridDelta_);

    for (int y = yStart; y <= yEnd; ++y) {
      for (int x = xStart; x <= xEnd; ++x) {

        double xReal = x * gridDelta_ - bounds_[0];
        double yReal = y * gridDelta_ - bounds_[2];

        double current = proximity.exposureAt(xReal, yReal);
        // Check if current is on the contour
        if (current == threshold) {
          IndexType pos;
          pos[0] = x;
          pos[1] = y;
          pointData.emplace_back(pos, 0.);
          break;
        }

        double minDist = std::numeric_limits<double>::max();
        int bestNx = -1, bestNy = -1;

        for (auto [dy, dx] : directions) {
          int nx = x + dx;
          int ny = y + dy;
          double nxReal = nx * gridDelta_ - bounds_[0];
          double nyReal = ny * gridDelta_ - bounds_[2];

          double neighbor = proximity.exposureAt(nxReal, nyReal);

          // Check if neighbor is on the contour
          // If so, skip checks and add neighbor when it becomes "current"
          if (neighbor == threshold)
            break;

          // Check if neighbor is on opposite side of the contour
          if ((current - threshold) * (neighbor - threshold) < 0) {
            // Interpolate sub-cell distance
            double dist =
                std::abs((threshold - current) / (neighbor - current));
            if (dist < minDist) {
              minDist = dist;
              bestNx = nx;
              bestNy = ny;
            }
          }
        }
        if ((minDist < 1.0) && (bestNx >= 0.) && (bestNy >= 0.)) {
          double sdfCurrent = minDist; // * gridDelta_;
          IndexType pos;
          pos[0] = x;
          pos[1] = y;
          double sign = (current < threshold) ? 1.0 : -1.0;
          pointData.emplace_back(pos, sign * sdfCurrent);
        }
      }
    }
    return pointData;
  }

  void addBlur(std::vector<NumericType> inSigmas,
               std::vector<NumericType> inWeights,
               NumericType inThreshold = 0.5, NumericType delta = -1.,
               int gridRefinement = 4) {
    weights = inWeights;
    threshold = inThreshold;
    beamDelta = (delta == -1.) ? gridDelta_ : delta;
    // Defines the ratio of the beam grid to the storage of the
    // illumination during Gaussians convolution
    gridRefinement = std::ceil(gridRefinement * beamDelta / gridDelta_);

    Logger::getInstance()
        .addInfo("gridDelta = " + std::to_string(gridDelta_) +
                 ", beamDelta = " + std::to_string(beamDelta) +
                 ", gridRefinement = " + std::to_string(gridRefinement))
        .print();

    // Scale sigmas to represent geometry dimensions
    NumericType exposureDelta = beamDelta / gridRefinement;
    for (auto sigma : inSigmas) {
      sigmas.push_back(sigma * gridDelta_ / exposureDelta);
    }
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

  std::size_t getNumberOfStructures() const { return structures.size(); }

private:
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
          Logger::getInstance()
              .addWarning("Missing referenced structure: " + sref.strName)
              .print();
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
    if constexpr (D == 3) {
      bounds_[4] = -1.;
      bounds_[5] = 1.;
    }
  }

  void addBox(lsDomainType2D layer2D, GDS::Element<NumericType> &element,
              const NumericType xOffset, const NumericType yOffset) const {

    assert(element.elementType == GDS::ElementType::elBox);
    assert(element.pointCloud.size() == 4); // GDSII box is a rectangle

    using VectorType = viennals::VectorType<NumericType, 2>;

    // The corners in GDS are typically ordered clockwise or counter-clockwise
    VectorType minCorner{
        std::min({element.pointCloud[0][0], element.pointCloud[1][0],
                  element.pointCloud[2][0], element.pointCloud[3][0]}),
        std::min({element.pointCloud[0][1], element.pointCloud[1][1],
                  element.pointCloud[2][1], element.pointCloud[3][1]})};

    VectorType maxCorner{
        std::max({element.pointCloud[0][0], element.pointCloud[1][0],
                  element.pointCloud[2][0], element.pointCloud[3][0]}),
        std::max({element.pointCloud[0][1], element.pointCloud[1][1],
                  element.pointCloud[2][1], element.pointCloud[3][1]})};

    // Generate a level set box using MakeGeometry
    viennals::MakeGeometry<NumericType, 2>(
        layer2D, viennals::Box<NumericType, 2>::New(minCorner, maxCorner))
        .apply();
  }

  void addPolygon(lsDomainType2D layer2D,
                  const GDS::Element<NumericType> &element,
                  const NumericType xOffset, const NumericType yOffset) {

    // Create a 2D level set from the polygon
    auto mesh = polygonToSurfaceMesh(element, xOffset, yOffset);
    lsDomainType2D tmpLS = lsDomainType2D::New(layer2D);
    viennals::FromSurfaceMesh<NumericType, 2>(tmpLS, mesh).apply();

    viennals::BooleanOperation<NumericType, 2>(
        layer2D, tmpLS, viennals::BooleanOperationEnum::UNION)
        .apply();
  }

  SmartPointer<viennals::Mesh<NumericType>>
  polygonToSurfaceMesh(const GDS::Element<NumericType> &element,
                       const NumericType xOffset, const NumericType yOffset) {
    auto mesh = viennals::Mesh<NumericType>::New();
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

  lsDomainType2D getMaskLevelSet(const int16_t layer, bool blurLayer = true) {

    const bool blurring = blurLayer && blur;
    auto GDSLevelSet = lsDomainType2D::New(bounds_, boundaryConds_,
                                           blurring ? beamDelta : gridDelta_);

    for (auto &str : structures) { // loop over all structures
      if (!str.isRef) {
        // add single layer
        if (auto contains = str.containsLayers.find(layer);
            contains != str.containsLayers.end()) {
          for (auto &el : str.elements) {
            if (el.layer == layer) {
              if (el.elementType == GDS::ElementType::elBox)
                addBox(GDSLevelSet, el, 0., 0.);
              else
                addPolygon(GDSLevelSet, el, 0., 0.);
            }
          }
        }

        for (auto &sref : str.sRefs) {
          auto refStr = getStructure(sref.strName);
          if (!refStr)
            continue;

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
              addBox(GDSLevelSet, elCopy, 0., 0.);
            else
              addPolygon(GDSLevelSet, elCopy, 0., 0.);
          }
        }
      }
    }

    if (blurring) {
      lsDomainType2D maskLS =
          lsDomainType2D::New(bounds_, boundaryConds_, gridDelta_);
      PointType pointData = applyBlur(GDSLevelSet);
      maskLS->insertPoints(pointData);
      maskLS->finalize(2);
      // viennals::Expand<double, 2>(maskLS, 2).apply();
      return maskLS;
    }
    return GDSLevelSet;
  }

private:
  std::vector<GDS::Structure<NumericType>> structures;
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
  NumericType beamDelta;
  int gridRefinement = 4;
};

} // namespace viennaps