#ifndef PS_NEAREST_NEIGHBORS_INTERPOLATION_HPP
#define PS_NEAREST_NEIGHBORS_INTERPOLATION_HPP

#include <array>
#include <cmath>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#include <psDataScaler.hpp>
#include <psKDTree.hpp>
#include <psSmartPointer.hpp>
#include <psValueEstimator.hpp>

// Class providing nearest neighbors interpolation
template <typename NumericType, int InputDim, int OutputDim,
          typename DataScaler =
              psStandardScaler<NumericType, InputDim, InputDim + OutputDim>,
          typename PointLocator =
              psKDTree<NumericType, InputDim, InputDim + OutputDim>>
class psNearestNeighborsInterpolation
    : public psValueEstimator<NumericType, InputDim, OutputDim, NumericType> {

  static_assert(std::is_base_of_v<
                psPointLocator<NumericType, InputDim, InputDim + OutputDim>,
                PointLocator>);

  static_assert(std::is_base_of_v<
                psDataScaler<NumericType, InputDim, InputDim + OutputDim>,
                DataScaler>);

  using Parent =
      psValueEstimator<NumericType, InputDim, OutputDim, NumericType>;

  using typename Parent::InputType;
  using typename Parent::OutputType;

  using Parent::DataDim;
  using Parent::dataSource;

  using DataPtr = typename decltype(dataSource)::element_type::DataPtr;
  using DataVector = std::vector<std::array<NumericType, DataDim>>;

  int numberOfNeighbors;
  PointLocator locator;
  NumericType distanceExponent;

  DataPtr data = nullptr;
  bool initialized = false;

public:
  psNearestNeighborsInterpolation()
      : numberOfNeighbors(3), distanceExponent(2.) {}

  psNearestNeighborsInterpolation(int passedNumberOfNeighbors,
                                  NumericType passedDistanceExponent = 2.)
      : numberOfNeighbors(passedNumberOfNeighbors),
        distanceExponent(passedDistanceExponent) {}

  void setNumberOfNeighbors(int passedNumberOfNeighbors) {
    numberOfNeighbors = passedNumberOfNeighbors;
  }

  void setDistanceExponent(NumericType passedDistanceExponent) {
    distanceExponent = passedDistanceExponent;
  }

  bool initialize() override {
    if (!dataSource)
      return false;

    data = dataSource->getAll();
    if (!data)
      return false;

    if (data->size() == 0)
      return false;

    DataScaler scaler;
    scaler.setData(data);
    scaler.apply();
    auto scalingFactors = scaler.getScalingFactors();

    locator.setPoints(*data);
    locator.setScalingFactors(scalingFactors);
    locator.build();

    initialized = true;
    return true;
  }

  std::tuple<OutputType, NumericType>
  estimate(const InputType &input) override {
    if (!initialized)
      if (!initialize())
        return {{}, {}};

    auto neighbors = locator.findKNearest(input, numberOfNeighbors);
    OutputType result{0};

    NumericType weightSum{0};
    NumericType minDistance{0};

    for (int j = 0; j < numberOfNeighbors; ++j) {
      auto [nearestIndex, distance] = neighbors->at(j);

      minDistance = std::min({distance, minDistance});

      NumericType weight;
      if (distance == 0) {
        for (int i = 0; i < OutputDim; ++i)
          result[i] = data->at(nearestIndex)[InputDim + i];
        weightSum = 1.;
        break;
      } else {
        weight = std::pow(1. / distance, distanceExponent);
      }
      for (int i = 0; i < OutputDim; ++i)
        result[i] += weight * data->at(nearestIndex)[InputDim + i];

      weightSum += weight;
    }

    for (int i = 0; i < OutputDim; ++i)
      result[i] /= weightSum;

    return {result, minDistance};
  }
};

#endif