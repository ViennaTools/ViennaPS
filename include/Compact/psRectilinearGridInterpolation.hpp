#ifndef PS_RECTILINEAR_GRID_INTERPOLATION_HPP
#define PS_RECTILINEAR_GRID_INTERPOLATION_HPP

#include <algorithm>
#include <array>
#include <set>
#include <vector>

#include <psValueEstimator.hpp>

// Class providing linear interpolation on rectilinear data grids
template <typename NumericType, int InputDim, int OutputDim>
class psRectilinearGridInterpolation
    : public psValueEstimator<NumericType, InputDim, OutputDim, bool> {

  using Parent = psValueEstimator<NumericType, InputDim, OutputDim, bool>;

  using typename Parent::InputType;
  using typename Parent::OutputType;

  using Parent::DataDim;
  using Parent::dataSource;

  using DataPtr = typename decltype(dataSource)::element_type::DataPtr;
  using DataVector = std::vector<std::array<NumericType, DataDim>>;

  DataPtr data = nullptr;

  std::array<std::set<NumericType>, InputDim> uniqueValues;
  bool initialized = false;

  // For rectilinear grid interpolation to work, we first have to ensure that
  // our input coordinates are arranged in a certain way
  // Future improvement: parallelize the recursive sorting using OpenMP taks
  bool rearrange(typename DataVector::iterator start,
                 typename DataVector::iterator end, int axis, bool capture) {
    bool equalSize = true;

    // We only reorder based on the input dimension, not the output dimension
    if (axis >= InputDim)
      return equalSize;

    size_t size = end - start;
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
            std::cout << "Data is not arranged in a rectilinear grid!\n";
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
    if (!dataSource)
      return false;

    data = dataSource->getAll();
    if (!data)
      return false;

    auto equalSize = rearrange(data->begin(), data->end(), 0, true);

    // #ifdef VIENNAPS_VERBOSE
    //     for (auto &ticks : uniqueValues) {
    //       std::cout << "Unique grid values: ";
    //       for (auto tick : ticks)
    //         std::cout << tick << ", ";
    //       std::cout << std::endl;
    //     }
    // #endif

    if (!equalSize) {
      std::cout << "Data is not arranged in a rectilinear grid!\n";
      return false;
    }

    for (int i = 0; i < InputDim; ++i)
      if (uniqueValues[i].empty()) {
        std::cout << "The grid has no values along dimension " << i
                  << std::endl;
        return false;
      }

    initialized = true;
    return true;
  }

  std::tuple<OutputType, bool> estimate(const InputType &input) override {
    if (!initialized)
      if (!initialize())
        return {{}, {}};

    bool isInside = true;
    for (int i = 0; i < InputDim; ++i)
      if (!uniqueValues[i].empty()) {
        // Check if the input lies within the bounds of our data grid
        if (input[i] < *(uniqueValues[i].begin()) ||
            input[i] > *(uniqueValues[i].rbegin())) {
          isInside = false;
        }
      } else {
        return {{}, {}};
      }

    std::array<size_t, InputDim> gridIndices;
    std::array<NumericType, InputDim> normalizedCoordinates;

    // Check in which hyperrectangle the provided input coordinates are located
    for (int i = 0; i < InputDim; ++i) {
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
    std::array<std::array<NumericType, OutputDim>, (1 << InputDim)>
        cornerValues;
    for (int i = 0; i < cornerValues.size(); ++i) {
      size_t index = 0;
      size_t stepsize = 1;

      for (int j = InputDim - 1; j >= 0; --j) {
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
      const auto &corner = data->at(index);
      std::copy(corner.cbegin() + InputDim, corner.cend(),
                cornerValues[i].begin());
    }

    // Now do the actual linear interpolation
    std::array<NumericType, OutputDim> result{0};
    for (int dim = 0; dim < OutputDim; ++dim) {
      for (int i = InputDim - 1; i >= 0; --i) {
        int stride = 1 << i;
        for (int j = 0; j < stride; ++j) {
          cornerValues[j][dim] =
              normalizedCoordinates[i] * cornerValues[j][dim] +
              (1 - normalizedCoordinates[i]) * cornerValues[j + stride][dim];
        }
      }
      result[dim] = cornerValues[0][dim];
    }

    return {result, isInside};
  }
};

#endif