#ifndef PS_RECTILINEAR_GRID_INTERPOLATION_HPP
#define PS_RECTILINEAR_GRID_INTERPOLATION_HPP

#include <algorithm>
#include <array>
#include <set>
#include <vector>

#include <psLogger.hpp>
#include <psValueEstimator.hpp>

// Class providing linear interpolation on rectilinear data grids
template <typename NumericType>
class psRectilinearGridInterpolation
    : public psValueEstimator<NumericType, bool> {

  using Parent = psValueEstimator<NumericType, bool>;

  using typename Parent::ItemType;
  using typename Parent::SizeType;
  using typename Parent::VectorType;

  using Parent::data;
  using Parent::dataChanged;
  using Parent::inputDim;
  using Parent::outputDim;

  std::vector<std::set<NumericType>> uniqueValues;
  VectorType localData;

  // For rectilinear grid interpolation to work, we first have to ensure that
  // our input coordinates are arranged in a certain way
  // Future improvement: parallelize the recursive sorting using OpenMP taks
  bool rearrange(typename VectorType::iterator start,
                 typename VectorType::iterator end, int axis, bool capture) {
    bool equalSize = true;

    // We only reorder based on the input dimension, not the output dimension
    if (axis >= inputDim)
      return equalSize;

    auto size = std::distance(start, end);
    if (size > 1) {
      // Now sort the data along the given axis
      std::sort(start, end, [&](const auto &a, const auto &b) {
        return a[axis] < b[axis];
      });

      // Record the indices that separate ranges of the same value
      size_t rangeSize = 0;
      std::vector<size_t> rangeBreaks;
      rangeBreaks.push_back(0);
      if (capture)
        uniqueValues[axis].insert(start->at(axis));

      for (unsigned i = 1; i < size; ++i)
        if ((start + i - 1)->at(axis) != (start + i)->at(axis)) {
          if (rangeSize == 0)
            rangeSize = i;

          size_t tmp = rangeBreaks.back();

          rangeBreaks.push_back(i);
          uniqueValues[axis].insert((start + i)->at(axis));

          if (rangeSize != i - tmp) {
            psLogger::getInstance()
                .addWarning("Data is not arranged in a rectilinear grid!")
                .print();
            equalSize = false;
            return false;
          }
        }

      rangeBreaks.push_back(size);

      // Recursively launch sorting tasks for each of the separated ranges. Only
      // the first leaf in each level of the tree is instructed to capture the
      // unique values.
      for (unsigned i = 1; i < rangeBreaks.size(); ++i)
        equalSize = equalSize && rearrange(start + rangeBreaks[i - 1],
                                           start + rangeBreaks[i], axis + 1,
                                           capture && (i == 1));
    }

    return equalSize;
  }

public:
  psRectilinearGridInterpolation() {}

  bool initialize() override {
    if (!data || (data && data->empty())) {
      psLogger::getInstance()
          .addWarning(
              "psRectilinearGridInterpolation: the provided data is empty.")
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

    localData.clear();
    localData.reserve(data->size());
    std::copy(data->begin(), data->end(), std::back_inserter(localData));

    uniqueValues.resize(inputDim);

    auto equalSize = rearrange(localData.begin(), localData.end(), 0, true);

    if (!equalSize) {
      psLogger::getInstance()
          .addWarning("Data is not arranged in a rectilinear grid!")
          .print();
      return false;
    }

    for (int i = 0; i < inputDim; ++i)
      if (uniqueValues[i].empty()) {
        psLogger::getInstance()
            .addWarning("The grid has no values along dimension " +
                        std::to_string(i))
            .print();
        return false;
      }

    dataChanged = false;
    return true;
  }

  std::optional<std::tuple<ItemType, bool>>
  estimate(const ItemType &input) override {
    if (dataChanged)
      if (!initialize())
        return {};

    bool isInside = true;
    for (int i = 0; i < inputDim; ++i)
      if (!uniqueValues[i].empty()) {
        // Check if the input lies within the bounds of our data grid
        if (input[i] < *(uniqueValues[i].begin()) ||
            input[i] > *(uniqueValues[i].rbegin())) {
          isInside = false;
        }
      } else {
        return {};
      }

    std::vector<SizeType> gridIndices(inputDim, 0);
    std::vector<NumericType> normalizedCoordinates(inputDim, 0.);

    // Check in which hyperrectangle the provided input coordinates are located
    for (int i = 0; i < inputDim; ++i) {
      if (input[i] <= *uniqueValues[i].begin()) {
        // The coordinate is lower than or equal to the lowest grid point along
        // the axis i.
        gridIndices[i] = 0;
        normalizedCoordinates[i] = 0;
      } else if (input[i] >= *uniqueValues[i].rbegin()) {
        // The coordinate is greater than or equal to the greatest grid point
        // along the axis i.
        gridIndices[i] = uniqueValues[i].size() - 1;
        normalizedCoordinates[i] = 1.;
      } else {
        // The corrdinate is somewhere in between (excluding) the lowest and
        // greatest grid point.

        // The following function returns an iterator pointing to the first
        // element that is greater than input[i].
        auto upperIt = uniqueValues[i].upper_bound(input[i]);

        // Get the index of the lower bound (upper bound index - 1)
        gridIndices[i] = std::distance(uniqueValues[i].begin(), upperIt) - 1;

        NumericType upperBound = *upperIt;
        NumericType lowerBound = *(--upperIt);
        normalizedCoordinates[i] =
            (input[i] - lowerBound) / (upperBound - lowerBound);
      }
    }

    // Now retrieve the values at the corners of the selected hyperrectangle
    std::vector<std::vector<NumericType>> cornerValues(
        1 << inputDim, std::vector<NumericType>(outputDim, 0.));

    for (int i = 0; i < cornerValues.size(); ++i) {
      size_t index = 0;
      size_t stepsize = 1;

      for (int j = inputDim - 1; j >= 0; --j) {
        // To get all combinations of corners of the hyperrectangle, we say
        // that each bit in the i variable corresponds to an axis. A zero
        // bit at a certain location represents a lower bound of the
        // hyperrectangle along the axis and a one represents the upper
        // bound.
        int lower = (i >> j) & 1;

        // If the grid index is at the maximum in this axis, always mark this
        // point as the lower point (thus if the input lies at or outside of the
        // upper edge boundary of the grid in the given axis, we use the same
        // corner multiple times)
        if (gridIndices[j] == uniqueValues[j].size() - 1)
          lower = 1;

        // If it is a lower bound, use the grid index itself, otherwise add
        // 1 to the index (= index of upper bound along that axis)
        index += (gridIndices[j] + (1 - lower)) * stepsize;
        stepsize *= uniqueValues[j].size();
      }
      const auto &corner = localData.at(index);
      std::copy(corner.cbegin() + inputDim, corner.cend(),
                cornerValues[i].begin());
    }

    // Now do the actual linear interpolation
    std::vector<NumericType> result(outputDim, 0.);
    for (int dim = 0; dim < outputDim; ++dim) {
      for (int i = inputDim - 1; i >= 0; --i) {
        int stride = 1 << i;
        for (int j = 0; j < stride; ++j) {
          cornerValues[j][dim] =
              normalizedCoordinates[i] * cornerValues[j][dim] +
              (1 - normalizedCoordinates[i]) * cornerValues[j + stride][dim];
        }
      }
      result[dim] = cornerValues[0][dim];
    }

    return {{result, isInside}};
  }
};

#endif