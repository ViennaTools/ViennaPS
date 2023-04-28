#ifndef PS_NEAREST_NEIGHBORS_INTERPOLATION_HPP
#define PS_NEAREST_NEIGHBORS_INTERPOLATION_HPP

#include <array>
#include <cmath>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

#include <psDataScaler.hpp>
#include <psKDTree.hpp>
#include <psSmartPointer.hpp>
#include <psValueEstimator.hpp>

template <typename VectorType, typename SizeType>
auto extractInputData(psSmartPointer<const VectorType> data, SizeType InputDim,
                      SizeType OutputDim) {
  VectorType inputData;
  using ElementType = typename VectorType::value_type;

  inputData.reserve(data->size());
  std::transform(data->begin(), data->end(), std::back_inserter(inputData),
                 [=](const auto &d) {
                   ElementType element;
                   std::copy_n(d.begin(), InputDim,
                               std::back_inserter(element));
                   return element;
                 });
  return inputData;
}

// Class providing nearest neighbors interpolation
template <typename NumericType,
          typename DataScaler = psStandardScaler<NumericType>>
class psNearestNeighborsInterpolation
    : public psValueEstimator<NumericType, NumericType> {

  static_assert(std::is_base_of_v<psDataScaler<NumericType>, DataScaler>,
                "psNearestNeighborsInterpolation: the provided DataScaler "
                "does not inherit from psDataScaler.");

  using Parent = psValueEstimator<NumericType, NumericType>;

  using typename Parent::ItemType;
  using typename Parent::SizeType;

  using Parent::data;
  using Parent::dataChanged;
  using Parent::inputDim;
  using Parent::outputDim;

  psKDTree<NumericType> kdtree;

  int numberOfNeighbors = 3.;
  NumericType distanceExponent = 2.;

public:
  psNearestNeighborsInterpolation() {}

  void setNumberOfNeighbors(int passedNumberOfNeighbors) {
    numberOfNeighbors = passedNumberOfNeighbors;
  }

  void setDistanceExponent(NumericType passedDistanceExponent) {
    distanceExponent = passedDistanceExponent;
  }

  bool initialize() override {
    if (!data || (data && data->empty())) {
      psLogger::getInstance()
          .addWarning(
              "psNearestNeighborsInterpolation: the provided data is empty.")
          .print();
      return false;
    }

    if (data->at(0).size() != inputDim + outputDim) {
      psLogger::getInstance()
          .addWarning(
              "psNearestNeighborsInterpolation: the sum of the provided "
              "InputDimension and OutputDimension does not match the "
              "dimension of the provided data.")
          .print();
      return false;
    }

    // Copy the first inputDim columns into a new vector
    auto inputData = extractInputData(data, inputDim, outputDim);

    DataScaler scaler(inputData);
    scaler.apply();
    auto scalingFactors = scaler.getScalingFactors();

    kdtree.setPoints(inputData, scalingFactors);
    kdtree.build();

    dataChanged = false;

    return true;
  }

  std::optional<std::tuple<ItemType, NumericType>>
  estimate(const ItemType &input) override {
    if (input.size() != inputDim) {
      psLogger::getInstance()
          .addWarning(
              "psNearestNeighborsInterpolation: No input data provided.")
          .print();
      return {};
    }

    if (dataChanged)
      if (!initialize())
        return {};

    auto neighborsOpt = kdtree.findKNearest(input, numberOfNeighbors);
    if (!neighborsOpt)
      return {};

    auto neighbors = neighborsOpt.value();

    ItemType result(outputDim, 0.);

    NumericType weightSum{0};
    NumericType minDistance = std::numeric_limits<NumericType>::infinity();

    for (int j = 0; j < numberOfNeighbors; ++j) {
      auto [nearestIndex, distance] = neighbors.at(j);

      minDistance = std::min({distance, minDistance});

      NumericType weight;
      if (distance == 0) {
        for (int i = 0; i < outputDim; ++i)
          result[i] = data->at(nearestIndex).at(inputDim + i);
        weightSum = 1.;
        break;
      } else {
        weight = std::pow(1. / distance, distanceExponent);
      }
      for (int i = 0; i < outputDim; ++i)
        result[i] += weight * data->at(nearestIndex).at(inputDim + i);

      weightSum += weight;
    }

    for (int i = 0; i < outputDim; ++i)
      result[i] /= weightSum;

    return {{result, minDistance}};
  }
};

#endif