#pragma once

#include "psProcessContext.hpp"

#include <lsMesh.hpp>

#include <vcKDTree.hpp>
#include <vcVectorType.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace viennaps {

using namespace viennacore;

template <class NumericType> struct PointCloud {
  std::vector<Vec3D<NumericType>> positions;
  std::vector<Vec3D<NumericType>> normals;

  [[nodiscard]] std::size_t size() const { return positions.size(); }

  void validate() const {
    if (positions.size() != normals.size()) {
      VIENNACORE_LOG_ERROR(
          "PointCloud: positions and normals must have the same size.");
    }
  }
};

template <class NumericType> class NeighborSearch {
public:
  using Neighbor = std::pair<std::size_t, double>;

  explicit NeighborSearch(const PointCloud<NumericType> &cloud)
      : cloud_(cloud) {
    cloud_.validate();
    points_.reserve(cloud_.positions.size());
    for (const auto &point : cloud_.positions) {
      points_.push_back(point);
    }
    tree_.setPoints(points_);
    tree_.build();
  }

  [[nodiscard]] std::vector<Neighbor> getKNN(std::size_t i, int k) const {
    if (i >= points_.size()) {
      VIENNACORE_LOG_ERROR("NeighborSearch::getKNN point index is invalid.");
    }
    const auto neighbors = tree_.findKNearest(points_[i], k);
    return convertNeighbors(neighbors);
  }

  [[nodiscard]] std::vector<Neighbor> getRadius(std::size_t i,
                                                double radius) const {
    if (i >= points_.size()) {
      VIENNACORE_LOG_ERROR("NeighborSearch::getRadius point index is invalid.");
    }
    const auto neighbors = tree_.findNearestWithinRadius(points_[i], radius);
    return convertNeighbors(neighbors);
  }

private:
  static std::vector<Neighbor> convertNeighbors(
      const std::optional<
          std::vector<std::pair<std::vector<double>::size_type, double>>>
          &neighbors) {
    std::vector<Neighbor> result;
    if (!neighbors) {
      return result;
    }
    result.reserve(neighbors->size());
    for (const auto &[index, distance] : *neighbors) {
      result.emplace_back(static_cast<std::size_t>(index), distance);
    }
    return result;
  }

  const PointCloud<NumericType> &cloud_;
  std::vector<Vec3D<NumericType>> points_;
  viennacore::KDTree<double, Vec3D<NumericType>> tree_;
};

template <class NumericType> class SurfaceDiffusionStencil {
public:
  using Parameters = SurfaceDiffusionParameters;
  using SparseRow = std::vector<std::pair<std::size_t, NumericType>>;

  explicit SurfaceDiffusionStencil(PointCloud<NumericType> cloud)
      : SurfaceDiffusionStencil(std::move(cloud), Parameters{}) {}

  SurfaceDiffusionStencil(PointCloud<NumericType> cloud, Parameters parameters)
      : cloud_(std::move(cloud)), parameters_(parameters) {
    cloud_.validate();
    build();
  }

  [[nodiscard]] const std::vector<SparseRow> &matrix() const { return L_; }

  [[nodiscard]] const SparseRow &row(std::size_t i) const { return L_.at(i); }

  [[nodiscard]] NumericType localScale(std::size_t i) const {
    return localScales_.at(i);
  }

private:
  struct DirectedEdge {
    std::size_t index = 0;
    NumericType weight = 0.;
  };

  struct EdgeAccumulator {
    NumericType weightSum = 0.;
    std::size_t count = 0;
  };

  void build() {
    NeighborSearch<NumericType> neighborSearch(cloud_);
    std::vector<std::vector<DirectedEdge>> directed(cloud_.size());
    localScales_.assign(cloud_.size(), NumericType(0.));

    for (std::size_t i = 0; i < cloud_.size(); ++i) {
      directed[i] = buildDirectedRow(i, neighborSearch);
    }

    L_.clear();
    L_.resize(cloud_.size());
    if (parameters_.symmetrizeWeights) {
      buildSymmetricRows(directed);
    } else {
      buildDirectedRows(directed);
    }
  }

  [[nodiscard]] std::vector<DirectedEdge>
  buildDirectedRow(std::size_t i,
                   const NeighborSearch<NumericType> &neighborSearch) {
    const auto neighbors =
        parameters_.radius > NumericType(0.)
            ? neighborSearch.getRadius(i, parameters_.radius)
            : neighborSearch.getKNN(i, parameters_.kNeighbors + 1);

    const Vec3D<NumericType> &normalI = cloud_.normals[i];
    const auto ni = Normalize(normalI);
    std::vector<std::pair<std::size_t, NumericType>> candidates;
    candidates.reserve(neighbors.size());

    NumericType distanceSum = 0.;
    std::size_t distanceCount = 0;
    for (const auto &[j, distance] : neighbors) {
      if (j == i || distance <= std::numeric_limits<NumericType>::epsilon()) {
        continue;
      }
      const Vec3D<NumericType> &normalJ = cloud_.normals[j];
      if (DotProduct(ni, Normalize(normalJ)) <= parameters_.normalCutoff) {
        continue;
      }
      candidates.push_back({j, static_cast<NumericType>(distance)});
      distanceSum += static_cast<NumericType>(distance);
      ++distanceCount;
    }

    if (distanceCount == 0) {
      return {};
    }

    const NumericType h =
        std::max(distanceSum / static_cast<NumericType>(distanceCount),
                 std::numeric_limits<NumericType>::epsilon());
    localScales_[i] = h;

    std::vector<DirectedEdge> row;
    row.reserve(candidates.size());
    for (const auto &[j, distance] : candidates) {
      const Vec3D<NumericType> &normalJ = cloud_.normals[j];
      const auto nj = Normalize(normalJ);
      const auto normalDot = DotProduct(ni, nj);
      const auto normalPenalty =
          (NumericType(1.) - normalDot * normalDot) /
          (parameters_.sigmaNormal * parameters_.sigmaNormal);
      NumericType weight =
          std::exp(-(distance * distance) / (h * h)) * std::exp(-normalPenalty);
      if (parameters_.normalizeByLocalScale) {
        weight /= h * h;
      }
      if (weight > std::numeric_limits<NumericType>::min()) {
        row.push_back({j, weight});
      }
    }
    return row;
  }

  void buildDirectedRows(const std::vector<std::vector<DirectedEdge>> &rows) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
      NumericType sum = 0.;
      L_[i].reserve(rows[i].size() + 1);
      for (const auto &edge : rows[i]) {
        L_[i].push_back({edge.index, edge.weight});
        sum += edge.weight;
      }
      L_[i].push_back({i, -sum});
    }
  }

  void buildSymmetricRows(const std::vector<std::vector<DirectedEdge>> &rows) {
    std::unordered_map<std::uint64_t, EdgeAccumulator> edgeWeights;
    edgeWeights.reserve(rows.size() * static_cast<std::size_t>(
                                          std::max(parameters_.kNeighbors, 1)));

    for (std::size_t i = 0; i < rows.size(); ++i) {
      for (const auto &edge : rows[i]) {
        if (edge.index == i) {
          continue;
        }
        const auto a = std::min(i, edge.index);
        const auto b = std::max(i, edge.index);
        auto &accumulator = edgeWeights[edgeKey(a, b)];
        accumulator.weightSum += edge.weight;
        ++accumulator.count;
      }
    }

    for (const auto &[key, accumulator] : edgeWeights) {
      const auto [i, j] = unpackEdgeKey(key);
      if (accumulator.count == 0 || i >= L_.size() || j >= L_.size()) {
        continue;
      }
      const auto weight =
          accumulator.weightSum / static_cast<NumericType>(accumulator.count);
      L_[i].push_back({j, weight});
      L_[j].push_back({i, weight});
    }

    for (std::size_t i = 0; i < L_.size(); ++i) {
      NumericType sum = 0.;
      for (const auto &[j, weight] : L_[i]) {
        (void)j;
        sum += weight;
      }
      L_[i].push_back({i, -sum});
    }
  }

  [[nodiscard]] static std::uint64_t edgeKey(std::size_t i, std::size_t j) {
    return (static_cast<std::uint64_t>(i) << 32) |
           static_cast<std::uint64_t>(j);
  }

  [[nodiscard]] static std::pair<std::size_t, std::size_t>
  unpackEdgeKey(std::uint64_t key) {
    return {static_cast<std::size_t>(key >> 32),
            static_cast<std::size_t>(key & 0xffffffffULL)};
  }

  PointCloud<NumericType> cloud_;
  Parameters parameters_;
  std::vector<SparseRow> L_;
  std::vector<NumericType> localScales_;
};

template <class NumericType> class SurfaceDiffusionSolver {
public:
  SurfaceDiffusionSolver() : isActive_(false) {}

  explicit SurfaceDiffusionSolver(SurfaceDiffusionStencil<NumericType> stencil)
      : stencil_(std::move(stencil)), isActive_(true) {}

  void setStencil(SurfaceDiffusionStencil<NumericType> stencil) {
    stencil_.emplace(std::move(stencil));
    isActive_ = true;
  }

  void setActive(bool active) { isActive_ = active; }

  bool isActive() const { return isActive_; }

  [[nodiscard]] std::vector<NumericType>
  applyLaplacian(const std::vector<NumericType> &u) const {
    if (!stencil_) {
      VIENNACORE_LOG_ERROR(
          "SurfaceDiffusionSolver::applyLaplacian called before stencil is "
          "set.");
      return {};
    }

    const auto &matrix = stencil_->matrix();
    if (u.size() != matrix.size()) {
      VIENNACORE_LOG_ERROR(
          "SurfaceDiffusionSolver::applyLaplacian vector size mismatch.");
    }

    std::vector<NumericType> result(u.size(), NumericType(0.));
#pragma omp parallel for
    for (std::size_t i = 0; i < matrix.size(); ++i) {
      for (const auto &[j, alpha] : matrix[i]) {
        result[i] += alpha * u[j];
      }
    }
    return result;
  }

  [[nodiscard]] std::vector<NumericType>
  stepExplicit(const std::vector<NumericType> &u, NumericType dt,
               NumericType diffusionCoefficient) const {
    if (!isActive_) {
      VIENNACORE_LOG_ERROR(
          "SurfaceDiffusionSolver::stepExplicit called before stencil is set.");
      return {};
    }

    auto laplacian = applyLaplacian(u);
    std::vector<NumericType> result(u.size(), NumericType(0.));
#pragma omp parallel for
    for (std::size_t i = 0; i < u.size(); ++i) {
      result[i] = u[i] + dt * diffusionCoefficient * laplacian[i];
    }
    return result;
  }

  [[nodiscard]] const SurfaceDiffusionStencil<NumericType> &stencil() const {
    return *stencil_;
  }

private:
  std::optional<SurfaceDiffusionStencil<NumericType>> stencil_;
  bool isActive_ = false;
};
} // namespace viennaps
