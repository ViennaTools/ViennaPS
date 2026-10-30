#pragma once

#include <lsMesh.hpp>
#include <vcKDTree.hpp>
#include <vcNFKDTree.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <numeric>
#include <utility>

namespace viennaps {

using namespace viennacore;

template <class NumericType, class MeshNT, class ResultType, bool d2 = true,
          bool d4 = true,
          class TreeType = NFKDTree<NumericType, Vec3D<NumericType>, 3>>
class ElementToPointData {
  std::vector<std::string> dataLabels_;
  SmartPointer<PointData<NumericType>> pointData_;
  SmartPointer<TreeType> elementKdTree_;
  SmartPointer<viennals::Mesh<NumericType>> diskMesh_;
  SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh_;
  NumericType conversionRadius_;
  bool alignedOnly_ = false;
  std::string postFix_ = "";

  struct CloseElements {
    std::vector<size_t> indices;
    std::vector<double> weights;
    // the weights are the coefficients of a fit per face, applied as they are
    bool fitted = false;
  };
  // fit per face (setFitPerFace): order, dimension of the surface and the
  // cosine of the largest angle between the normals of a point and of the
  // elements of its face
  int fitOrder_ = 0;
  int dimension_ = 3;
  double fitCosine_ = 1.;

  std::vector<std::vector<ResultType>> elementDataArrays_;
  std::vector<CloseElements> closeElements_;
  static constexpr bool discard2 = d2;
  static constexpr bool discard4 = d4;
  static constexpr double weightThreshold = 1e-4;

public:
  ElementToPointData() = default;

  ElementToPointData(
      const std::vector<std::string> &dataLabels,
      SmartPointer<PointData<NumericType>> pointData, // target point data
      SmartPointer<TreeType> elementKdTree,
      SmartPointer<viennals::Mesh<NumericType>> diskMesh,
      SmartPointer<viennals::Mesh<MeshNT>> surfMesh,
      const NumericType conversionRadius)
      : dataLabels_(dataLabels), pointData_(pointData),
        elementKdTree_(elementKdTree), diskMesh_(diskMesh),
        surfaceMesh_(surfMesh), conversionRadius_(conversionRadius) {}

  void
  setElementDataArrays(std::vector<std::vector<ResultType>> elementDataArrays) {
    elementDataArrays_ = std::move(elementDataArrays);
  }

  void setDataLabels(std::vector<std::string> dataLabels) {
    dataLabels_ = std::move(dataLabels);
  }

  void setPointData(SmartPointer<PointData<NumericType>> pointData) {
    pointData_ = pointData;
  }

  void setConversionRadius(NumericType radius) { conversionRadius_ = radius; }

  /// Whether only elements whose normal points to the same side as the
  /// point's normal contribute to it, as in the smoothing of the disk engine,
  /// so that a point next to a sharp edge does not take the flux of the face
  /// across it. Default false.
  void setAlignedElementsOnly(bool alignedOnly) { alignedOnly_ = alignedOnly; }

  /// Each point takes the value at its position of a least-squares fit in
  /// its tangent plane over the elements of its face, those within the
  /// conversion radius whose normal lies within angle (radians) of the
  /// point's, weighted by area and by distance; order 1 is linear, 2
  /// quadratic, 0 off. Averaging takes the flux of the middle of a face next
  /// to its end and flattens a curved profile; the fit follows both. Points
  /// with too few elements on their face keep the average. Dimension 2 fits
  /// along the line of a 2D surface.
  void setFitPerFace(int order, int dimension, double angle) {
    fitOrder_ = order;
    dimension_ = dimension;
    fitCosine_ = std::cos(angle);
  }

  void setDiskMesh(SmartPointer<viennals::Mesh<NumericType>> diskMesh) {
    diskMesh_ = diskMesh;
  }

  void setSurfaceMesh(SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh) {
    surfaceMesh_ = surfaceMesh;
  }

  void setElementKdTree(SmartPointer<TreeType> elementKdTree) {
    elementKdTree_ = elementKdTree;
  }

  void setPostFix(const std::string &postFix) { postFix_ = postFix; }

  void prepare(bool skipBuild = false) {
    const auto numData = dataLabels_.size();
    const auto &points = diskMesh_->nodes;
    const auto numPoints = points.size();
    const auto numElements = elementKdTree_->getNumberOfPoints();
    const auto normals = diskMesh_->cellData.getVectorData("Normals");
    const auto elementNormals = surfaceMesh_->cellData.getVectorData("Normals");
    assert(normals && elementNormals);
    assert(elementNormals->size() == numElements);

    // prepare point data container
    pointData_->clear();
    for (const auto &label : dataLabels_) {
      std::vector<NumericType> data(numPoints, 0.);
      pointData_->insertNextScalarData(std::move(data), label + postFix_);
    }

    if (skipBuild)
      return;

    closeElements_.clear();
    closeElements_.resize(numPoints);
    const auto lookupRadiusSquared_ = conversionRadius_ * conversionRadius_;

#pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < numPoints; i++) {

      // we have to use the squared distance here
      auto closeElements =
          elementKdTree_
              ->findNearestWithinRadius(points[i], lookupRadiusSquared_)
              .value();

      std::vector<double> weights;
      std::vector<size_t> elementIndices;
      weights.reserve(closeElements.size());
      elementIndices.reserve(closeElements.size());

      // compute weights based on normal alignment
      unsigned numClosePoints = 0;
      for (std::size_t n = 0; n < closeElements.size(); ++n) {
        const auto &[idx, distanceSquared] = closeElements[n];
        assert(idx < numElements);
        assert(distanceSquared < lookupRadiusSquared_ + 1e-6);

        const double alignment =
            DotProductNT(normals->at(i), elementNormals->at(idx));
        if (alignedOnly_ && !(alignment > 0.))
          continue;
        const double weight =
            alignment +
            (lookupRadiusSquared_ - distanceSquared) / lookupRadiusSquared_;

        if (!std::isnan(weight) && weight > weightThreshold) {
          weights.push_back(weight);
          elementIndices.push_back(idx);
        }
      }

      assert(!weights.empty() && !elementIndices.empty() &&
             "Weights and element indices should not be empty here.");
      closeElements_[i].indices = std::move(elementIndices);
      closeElements_[i].weights = std::move(weights);
      closeElements_[i].fitted = false;
      if (fitOrder_ > 0)
        fitPerFace(i, closeElements, normals->at(i), *elementNormals,
                   lookupRadiusSquared_);
    }
  }

  // the coefficients of the fit per face of point i where the elements of
  // its face determine it: the value at the point, the first coefficient of
  // the weighted least-squares polynomial in tangent coordinates scaled by
  // the radius, is a weighted sum of the element values
  template <class Close, class Normal, class ElementNormals>
  void fitPerFace(unsigned i, const Close &closeElements, const Normal &normal,
                  const ElementNormals &elementNormals, double radius2) {
    const auto &p = diskMesh_->nodes[i];
    const double n[3] = {double(normal[0]), double(normal[1]),
                         double(normal[2])};
    // tangent directions; in 2D the first lies in the plane of the surface
    double a[3] = {1., 0., 0.};
    if (dimension_ == 2) {
      a[0] = 0.;
      a[2] = 1.;
    } else if (std::abs(n[0]) > 0.9) {
      a[0] = 0.;
      a[1] = 1.;
    }
    double t1[3] = {a[1] * n[2] - a[2] * n[1], a[2] * n[0] - a[0] * n[2],
                    a[0] * n[1] - a[1] * n[0]};
    const double length =
        std::sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
    if (length < 1e-12)
      return;
    for (auto &c : t1)
      c /= length;
    const double t2[3] = {n[1] * t1[2] - n[2] * t1[1],
                          n[2] * t1[0] - n[0] * t1[2],
                          n[0] * t1[1] - n[1] * t1[0]};
    const double radius = std::sqrt(radius2);
    const int terms =
        (dimension_ == 2) ? fitOrder_ + 1 : (fitOrder_ == 1 ? 3 : 6);
    std::vector<size_t> indices;
    std::vector<std::array<double, 6>> rows;
    std::vector<double> w;
    for (const auto &[idx, distanceSquared] : closeElements) {
      if (DotProductNT(normal, elementNormals.at(idx)) < fitCosine_)
        continue;
      const auto &tri = surfaceMesh_->triangles[idx];
      const auto &q0 = surfaceMesh_->nodes[tri[0]],
                 &q1 = surfaceMesh_->nodes[tri[1]],
                 &q2 = surfaceMesh_->nodes[tri[2]];
      double d[3], e1[3], e2[3];
      for (int k = 0; k < 3; ++k) {
        d[k] = (double(q0[k]) + double(q1[k]) + double(q2[k])) / 3. - p[k];
        e1[k] = double(q1[k]) - double(q0[k]);
        e2[k] = double(q2[k]) - double(q0[k]);
      }
      const double cross[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                               e1[2] * e2[0] - e1[0] * e2[2],
                               e1[0] * e2[1] - e1[1] * e2[0]};
      const double area =
          0.5 * std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] +
                          cross[2] * cross[2]);
      const double u = (d[0] * t1[0] + d[1] * t1[1] + d[2] * t1[2]) / radius;
      const double v = (d[0] * t2[0] + d[1] * t2[1] + d[2] * t2[2]) / radius;
      std::array<double, 6> r{};
      if (dimension_ == 2)
        r = {1., u, u * u, 0., 0., 0.};
      else
        r = {1., u, v, u * u, u * v, v * v};
      indices.push_back(idx);
      rows.push_back(r);
      w.push_back(area * std::max(0., (radius2 - distanceSquared) / radius2));
    }
    if (int(indices.size()) < terms + 1)
      return;
    // normal equations, inverted by Gauss-Jordan elimination with partial
    // pivoting; a point whose elements leave them singular keeps the average
    double M[6][6] = {}, inverse[6][6] = {};
    for (std::size_t k = 0; k < rows.size(); ++k)
      for (int r = 0; r < terms; ++r)
        for (int c = 0; c < terms; ++c)
          M[r][c] += w[k] * rows[k][r] * rows[k][c];
    double scale = 0.;
    for (int r = 0; r < terms; ++r) {
      inverse[r][r] = 1.;
      scale = std::max(scale, std::abs(M[r][r]));
    }
    for (int col = 0; col < terms; ++col) {
      int pivot = col;
      for (int r = col + 1; r < terms; ++r)
        if (std::abs(M[r][col]) > std::abs(M[pivot][col]))
          pivot = r;
      if (!(std::abs(M[pivot][col]) > 1e-6 * scale))
        return;
      if (pivot != col)
        for (int c = 0; c < terms; ++c) {
          std::swap(M[pivot][c], M[col][c]);
          std::swap(inverse[pivot][c], inverse[col][c]);
        }
      const double diagonal = M[col][col];
      for (int c = 0; c < terms; ++c) {
        M[col][c] /= diagonal;
        inverse[col][c] /= diagonal;
      }
      for (int r = 0; r < terms; ++r)
        if (r != col) {
          const double f = M[r][col];
          for (int c = 0; c < terms; ++c) {
            M[r][c] -= f * M[col][c];
            inverse[r][c] -= f * inverse[col][c];
          }
        }
    }
    std::vector<double> coefficients(indices.size());
    for (std::size_t k = 0; k < indices.size(); ++k) {
      double c = 0.;
      for (int b = 0; b < terms; ++b)
        c += inverse[0][b] * rows[k][b];
      coefficients[k] = w[k] * c;
    }
    closeElements_[i].indices = std::move(indices);
    closeElements_[i].weights = std::move(coefficients);
    closeElements_[i].fitted = true;
  }

  void convert() {
    const auto numData = dataLabels_.size();
    const auto &points = diskMesh_->nodes;
    const auto numPoints = points.size();

    if (elementDataArrays_.size() != numData) {
      VIENNACORE_LOG_ERROR(
          "ElementToPointData: "
          "Number of data arrays does not match expected count.");
    }

#pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < numPoints; i++) {

      const auto &[indices, weights, fitted] = closeElements_[i];

      for (unsigned j = 0; j < numData; ++j) {
        NumericType value = NumericType(0);
        const auto &elementData = elementDataArrays_[j];
        auto pointData = pointData_->getScalarData(j);

        // the fit per face, a flux of no element below zero
        if (fitted) {
          double fit = 0.;
          for (size_t k = 0; k < indices.size(); ++k)
            fit += weights[k] * elementData[indices[k]];
          pointData->at(i) = NumericType(std::max(0., fit));
          continue;
        }

        // Discard outlier values if enabled
        const auto weightsCleaned =
            discardOutliers(weights, indices, elementData);

        // Compute weighted average
        const double sum = std::accumulate(weightsCleaned.cbegin(),
                                           weightsCleaned.cend(), 0.0);

        if (sum > 1e-6) [[likely]] {
          for (size_t k = 0; k < indices.size(); ++k) {
            value += weightsCleaned[k] * elementData[indices[k]];
          }
          value /= sum;
        } else {
          // Fallback if all weights were discarded
          auto nearestPoint = elementKdTree_->findNearest(points[i]);
          value = elementData[nearestPoint->first];
        }

        pointData->at(i) = value;
      }
    }
  }

  void apply() {
    validate();
    prepare();
    convert();
  }

  bool validate() const {
    if (!pointData_) {
      VIENNACORE_LOG_ERROR("ElementToPointData: PointData pointer is not set.");
      return false;
    }
    if (!elementKdTree_) {
      VIENNACORE_LOG_ERROR(
          "ElementToPointData: Element KDTree pointer is not set.");
      return false;
    }
    if (!diskMesh_) {
      VIENNACORE_LOG_ERROR("ElementToPointData: Disk mesh pointer is not set.");
      return false;
    }
    if (!surfaceMesh_) {
      VIENNACORE_LOG_ERROR(
          "ElementToPointData: Surface mesh pointer is not set.");
      return false;
    }
    if (dataLabels_.empty()) {
      VIENNACORE_LOG_ERROR("ElementToPointData: No data labels provided.");
      return false;
    }
    if (elementDataArrays_.size() != dataLabels_.size()) {
      VIENNACORE_LOG_ERROR(
          "ElementToPointData: Number of element data arrays does not match "
          "number of data labels.");
      return false;
    }
    return true;
  }

private:
  // Helper function to find min/max values and their indices
  struct MinMaxInfo {
    NumericType min1, min2, max1, max2;
    int minIdx1, minIdx2, maxIdx1, maxIdx2;

    MinMaxInfo()
        : min1(std::numeric_limits<NumericType>::max()),
          min2(std::numeric_limits<NumericType>::max()), max1(0), max2(0),
          minIdx1(-1), minIdx2(-1), maxIdx1(-1), maxIdx2(-1) {}
  };

  static MinMaxInfo
  findMinMaxValues(const std::vector<size_t> &closePoints,
                   const std::vector<ResultType> &elementData) {
    MinMaxInfo info;

    for (std::size_t k = 0; k < closePoints.size(); ++k) {
      const auto value = elementData[closePoints[k]];

      // Update min values
      if (value < info.min1) {
        info.min2 = info.min1;
        info.minIdx2 = info.minIdx1;
        info.min1 = value;
        info.minIdx1 = static_cast<int>(k);
      } else if (value < info.min2) {
        info.min2 = value;
        info.minIdx2 = static_cast<int>(k);
      }

      // Update max values
      if (value > info.max1) {
        info.max2 = info.max1;
        info.maxIdx2 = info.maxIdx1;
        info.max1 = value;
        info.maxIdx1 = static_cast<int>(k);
      } else if (value > info.max2) {
        info.max2 = value;
        info.maxIdx2 = static_cast<int>(k);
      }
    }

    return info;
  }

  static auto discardOutliers(const std::vector<double> &weights,
                              const std::vector<size_t> &closePoints,
                              const std::vector<ResultType> &elementData) {

    // copy weights to modify
    auto weightsCopy = weights;

    if (discard4 && closePoints.size() > 4) {
      const auto info = findMinMaxValues(closePoints, elementData);

      if (info.maxIdx1 != -1 && info.maxIdx2 != -1 && info.minIdx1 != -1 &&
          info.minIdx2 != -1) {
        // settting to 0 == discarding the weight
        weightsCopy[info.minIdx1] = 0.0;
        weightsCopy[info.minIdx2] = 0.0;
        weightsCopy[info.maxIdx1] = 0.0;
        weightsCopy[info.maxIdx2] = 0.0;
      }
    } else if (discard2 && closePoints.size() > 2) {
      const auto info = findMinMaxValues(closePoints, elementData);

      if (info.minIdx1 != -1 && info.maxIdx1 != -1) {
        weightsCopy[info.minIdx1] = 0.0;
        weightsCopy[info.maxIdx1] = 0.0;
      }
    }

    return weightsCopy;
  }

  template <class AT, class BT>
  static double DotProductNT(const std::array<AT, 3> &pVecA,
                             const std::array<BT, 3> &pVecB) {
    double dot = 0.0;
    for (size_t i = 0; i < 3; ++i)
      dot += pVecA[i] * pVecB[i];
    return dot;
  }
};

} // namespace viennaps
