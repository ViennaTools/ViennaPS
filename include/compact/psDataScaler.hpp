#ifndef PS_DATA_SCALER_HPP
#define PS_DATA_SCALER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

#include <psSmartPointer.hpp>

// Base class for data scalers
template <typename NumericType> class psDataScaler {
protected:
  using ItemType = std::vector<NumericType>;
  using ItemVectorType = std::vector<ItemType>;

  ItemType scalingFactors{};

public:
  virtual void apply() = 0;

  ItemType getScalingFactors() const { return scalingFactors; }
};

// Class that calculates scaling factors based on standard deviation
template <typename NumericType>
class psStandardScaler : public psDataScaler<NumericType> {
  using Parent = psDataScaler<NumericType>;

  using typename Parent::ItemType;
  using typename Parent::ItemVectorType;

  using Parent::scalingFactors;

  const ItemVectorType &data;

public:
  psStandardScaler(const ItemVectorType &passedData) : data(passedData) {}

  void apply() override {
    if (data.empty())
      return;

    const auto &dat = data;

    int D = data[0].size();
    std::vector<NumericType> mean(D, 0.);
    scalingFactors.resize(D, 0.);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < D; ++i)
      mean[i] = std::accumulate(dat.begin(), dat.end(), 0.,
                                [&](const auto &sum, const auto &element) {
                                  return sum + element.at(i);
                                });
    for (int i = 0; i < D; ++i)
      mean[i] /= dat.size();

    std::vector<NumericType> stddev(D, 0.);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < D; ++i)
      stddev[i] = std::accumulate(dat.begin(), dat.end(), 0.,
                                  [&](const auto &sum, const auto &element) {
                                    return sum + (element[i] - mean[i]) *
                                                     (element[i] - mean[i]);
                                  });

    for (int i = 0; i < D; ++i) {
      stddev[i] = std::sqrt(stddev[i] / dat.size());
      if (stddev[i] > 0)
        scalingFactors[i] = 1.0 / stddev[i];
      else
        scalingFactors[i] = 1.0;
    }
  }
};

// Class that calculates scaling factors based on median distances
template <typename NumericType>
class psMedianDistanceScaler : public psDataScaler<NumericType> {
  using Parent = psDataScaler<NumericType>;

  using typename Parent::ItemType;
  using typename Parent::ItemVectorType;

  using Parent::scalingFactors;

  const ItemVectorType &data;

public:
  psMedianDistanceScaler(const ItemVectorType &passedData) : data(passedData) {}

  void apply() override {
    if (data.empty())
      return;

    size_t size = data.size();
    size_t triSize = size * (size - 1) / 2;

    const auto &dat = data;

    int D = dat[0].size();
    scalingFactors.resize(D, 0.);

    std::vector<NumericType> distances(triSize, 0.);
    for (int i = 0; i < D; ++i) {
      {
#pragma omp parallel for default(none) firstprivate(i, size)                   \
    shared(dat, distances) schedule(dynamic)
        for (int j = 1; j < size; ++j) {
          for (int k = 0; k < j; ++k)
            distances[j * (j - 1) / 2 + k] =
                std::abs(dat.at(j)[i] - dat.at(k)[i]);
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