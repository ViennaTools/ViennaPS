#ifndef PS_POINT_LOCATOR_HPP
#define PS_POINT_LOCATOR_HPP

#include <array>
#include <utility> // std::pair
#include <vector>  // std::vector

#include <psSmartPointer.hpp>

template <unsigned int x, int axis>
inline constexpr unsigned int recursiveBitSum() {
  constexpr unsigned int val = ((x >> (axis - 1)) & 1);
  if constexpr (axis > 1)
    return val + recursiveBitSum<x, axis - 1>();
  else
    return val;
}

template <size_t N, const bool (&bitArray)[N], int axis = N>
inline constexpr unsigned int bitArrayToMask() {
  constexpr unsigned int mask = (1 << (N - axis)) * bitArray[axis - 1];
  if constexpr (axis > 1) {
    return mask + bitArrayToMask<N, bitArray, axis - 1>();
  } else {
    return (1 << (N - 1)) * bitArray[0];
  }
}

template <int D, int N, int Axis = D>
inline constexpr unsigned int maskNLower() {
    constexpr int index = D - Axis;
    constexpr unsigned int val = (index < N) ? (1 << index) : 0;
    if constexpr (index < N)
        return val | maskNLower<D, N, Axis - 1>();
    else
        return val;
}

template <class NumericType, int D, unsigned int axisMask = -1U>
struct psPointLocator {
  static constexpr unsigned int usedDimensions = recursiveBitSum<axisMask, D>();
  static_assert(
      usedDimensions >= 1,
      "The provided axes exclude mask would mask out all axes, but at "
      "least one axis must remain unmask for the tree to work "
      "properly.");

  using VectorType = std::array<NumericType, D>;
  using PointType = std::array<NumericType, usedDimensions>;
  using SizeType = std::size_t;

  virtual void build() = 0;

  virtual void setPoints(std::vector<VectorType> &passedPoints) = 0;

  virtual void setScalingFactors(const PointType &passedScalingFactors) = 0;

  virtual std::pair<SizeType, NumericType>
  findNearest(const PointType &x) const = 0;

  virtual psSmartPointer<std::vector<std::pair<SizeType, NumericType>>>
  findKNearest(const PointType &x, const int k) const = 0;

  virtual psSmartPointer<std::vector<std::pair<SizeType, NumericType>>>
  findNearestWithinRadius(const PointType &x,
                          const NumericType radius) const = 0;
};
#endif