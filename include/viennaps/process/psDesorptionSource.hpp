#pragma once

#include <rayReflection.hpp>
#include <raySource.hpp>
#include <vcRNG.hpp>
#include <vcVectorType.hpp>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace viennaps {

using namespace viennacore;

template <typename NumericType> struct DiskDesorptionSourceData {
  using VecType = Vec3D<NumericType>;

  std::vector<VecType> positions;
  std::vector<VecType> normals;
  std::vector<NumericType> weights;
  NumericType sourceArea = 0.;
  NumericType sourceOffset = 0.;
  bool hasSource = false;
};

template <typename OutputNumericType, typename InputNumericType, int D>
DiskDesorptionSourceData<OutputNumericType> makeDiskDesorptionSourceData(
    const std::vector<Vec3D<InputNumericType>> &positions,
    const std::vector<Vec3D<InputNumericType>> &normals,
    const std::vector<InputNumericType> &weights, InputNumericType gridDelta,
    InputNumericType diskRadius, bool useLineAreaIn2D = false) {
  DiskDesorptionSourceData<OutputNumericType> data;

  if (positions.size() != normals.size() || positions.size() != weights.size())
    return data;

  data.positions.resize(positions.size());
  data.normals.resize(normals.size());
  data.weights.resize(weights.size());
  data.sourceOffset = static_cast<OutputNumericType>(gridDelta * 1e-4);

#pragma omp parallel for
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(positions.size());
       ++i) {
    const auto idx = static_cast<std::size_t>(i);
    data.positions[idx] = {
        static_cast<OutputNumericType>(positions[idx][0]),
        static_cast<OutputNumericType>(positions[idx][1]),
        static_cast<OutputNumericType>(positions[idx][2])};
    data.normals[idx] = {static_cast<OutputNumericType>(normals[idx][0]),
                         static_cast<OutputNumericType>(normals[idx][1]),
                         static_cast<OutputNumericType>(normals[idx][2])};
    data.weights[idx] = static_cast<OutputNumericType>(weights[idx]);
  }

  for (auto weight : data.weights) {
    if (weight > OutputNumericType(0.)) {
      data.hasSource = true;
      break;
    }
  }

  constexpr OutputNumericType pi =
      static_cast<OutputNumericType>(3.14159265358979323846);
  const auto radius = static_cast<OutputNumericType>(diskRadius);
  OutputNumericType areaPerSource = radius * radius * pi;
  if constexpr (D == 2) {
    if (useLineAreaIn2D)
      areaPerSource = OutputNumericType(2.) * radius;
  }
  data.sourceArea =
      static_cast<OutputNumericType>(positions.size()) * areaPerSource;

  return data;
}

// Viennaray source that emits rays from surface disk positions with initial
// weights proportional to the local desorption rate r_des * N_A (Eq. 1,
// Panagopoulos & Lill 2023). Each disk i emits cosine-distributed rays with
// weight d_i = k_des * theta_i * Gamma_s * N_A / Gamma_source. Coverage is
// the sole input — the incoming flux plays no role here.
template <typename NumericType, int D>
class DesorptionSource : public viennaray::Source<NumericType> {
  using VecType = Vec3D<NumericType>;

  const std::vector<VecType> positions_;
  const std::vector<VecType> normals_;
  const std::vector<NumericType> weights_; // d_i per disk
  NumericType sourceArea_;
  size_t raysPerPoint_;
  NumericType offset_; // small normal offset to avoid self-intersection

public:
  DesorptionSource(DiskDesorptionSourceData<NumericType> sourceData,
                   size_t raysPerPoint)
      : positions_(std::move(sourceData.positions)),
        normals_(std::move(sourceData.normals)),
        weights_(std::move(sourceData.weights)),
        sourceArea_(sourceData.sourceArea), raysPerPoint_(raysPerPoint),
        offset_(sourceData.sourceOffset) {}

  DesorptionSource(std::vector<VecType> positions,
                   std::vector<VecType> normals,
                   std::vector<NumericType> weights, NumericType gridDelta,
                   NumericType diskRadius, size_t raysPerPoint)
      : DesorptionSource(
            makeDiskDesorptionSourceData<NumericType, NumericType, D>(
                positions, normals, weights, gridDelta, diskRadius),
            raysPerPoint) {}

  std::array<VecType, 2> getOriginAndDirection(size_t idx,
                                                RNG &rng) const override {
    const size_t diskIdx = idx / raysPerPoint_;
    const auto &pos = positions_[diskIdx];
    const auto &norm = normals_[diskIdx];
    VecType origin = pos + norm * offset_;
    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(norm, rng);
    return {origin, direction};
  }

  [[nodiscard]] size_t getNumPoints() const override {
    return positions_.size();
  }

  [[nodiscard]] NumericType getSourceArea() const override {
    return sourceArea_;
  }

  [[nodiscard]] NumericType getInitialRayWeight(size_t idx) const override {
    return weights_[idx / raysPerPoint_];
  }
};

} // namespace viennaps
