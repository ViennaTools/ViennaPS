#ifndef PS_DATA_SCALER_HPP
#define PS_DATA_SCALER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <psSmartPointer.hpp>

// Base class for data scalers
template <typename NumericType, int D, int Dim> class psDataScaler {
  static_assert(D <= Dim);

protected:
  using VectorType = std::array<NumericType, Dim>;

  psSmartPointer<std::vector<VectorType>> data = nullptr;
  std::array<NumericType, D> scalingFactors;

public:
  void setData(psSmartPointer<std::vector<VectorType>> passedData) {
    data = passedData;
  }

  virtual void apply() = 0;

  std::array<NumericType, D> getScalingFactors() const {
    return scalingFactors;
  }
};

// Class that calculates scaling factors based on standard deviation
template <typename NumericType, int D, int Dim>
class psStandardScaler : public psDataScaler<NumericType, D, Dim> {
  using Parent = psDataScaler<NumericType, D, Dim>;

  using Parent::data;
  using Parent::scalingFactors;

public:
  psStandardScaler() {}

  void apply() override {
    if (!data)
      return;

    std::vector<NumericType> mean(D, 0.);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < D; ++i)
      mean[i] = std::accumulate(data->begin(), data->end(), 0.,
                                [&](const auto &sum, const auto &element) {
                                  return sum + element[i];
                                });
    for (int i = 0; i < D; ++i)
      mean[i] /= data->size();

    std::vector<NumericType> stddev(D, 0.);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < D; ++i)
      stddev[i] = std::accumulate(data->begin(), data->end(), 0.,
                                  [&](const auto &sum, const auto &element) {
                                    return sum + (element[i] - mean[i]) *
                                                     (element[i] - mean[i]);
                                  });

    for (int i = 0; i < D; ++i) {
      stddev[i] = std::sqrt(stddev[i] / data->size());
      if (stddev[i] > 0)
        scalingFactors[i] = 1.0 / stddev[i];
      else
        scalingFactors[i] = 1.0;
    }
  }
};

// Class that calculates scaling factors based on median distances
template <typename NumericType, int D, int Dim>
class psMedianDistanceScaler : public psDataScaler<NumericType, D, Dim> {
  using Parent = psDataScaler<NumericType, D, Dim>;

  using Parent::data;
  using Parent::scalingFactors;

public:
  psMedianDistanceScaler() {}

  void apply() override {
    if (!data)
      return;

    size_t size = data->size();
    size_t triSize = size * (size - 1) / 2;

    auto dat = data;

    std::vector<NumericType> distances(triSize);
    for (int i = 0; i < D; ++i) {
      {
#pragma omp parallel for default(none) firstprivate(i, size)                   \
    shared(dat, distances) schedule(dynamic)
        for (int j = 1; j < size; ++j) {
          for (int k = 0; k < j; ++k)
            distances[j * (j - 1) / 2 + k] =
                std::abs(dat->at(j)[i] - dat->at(k)[i]);
        }

        size_t medianIndex = triSize / 2;
        std::nth_element(distances.begin(), distances.begin() + medianIndex,
                         distances.end());
        if (distances[medianIndex] > 0)
          scalingFactors[i] = 1.0 / distances[medianIndex];
        else
          scalingFactors[i] = 1.0;
      }
    }
  }
};

#endif