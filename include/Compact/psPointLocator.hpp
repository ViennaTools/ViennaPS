#ifndef PS_POINT_LOCATOR_HPP
#define PS_POINT_LOCATOR_HPP

#include <array>
#include <utility> // std::pair
#include <vector>  // std::vector

#include <psSmartPointer.hpp>

/**
 * Calculates the sum of the bits of a given unsigned integer value `X` in a
 * recursive manner. The parameter `Axis` specifies the number of bits to
 * consider, with `Axis` equal to the number of bits in `X` giving the full sum
 * of all bits in `X`. The function can be called with different values of
 * `Axis` to sum a subset of the bits in `X`. The function returns the sum as an
 * unsigned integer value.
 *
 * Example usage:
 *  recursiveBitSum<11, 4>() // returns 3, as the sum of the bits in the binary
 *  representation of 11 (1011) is 3 recursiveBitSum<11, 3>() // returns 2, as
 *  the sum of the first 3 bits in the binary representation of 11 (101) is 2
 */
template <unsigned int X, int Axis>
inline constexpr unsigned int recursiveBitSum() {
  // Calculate the value of the current bit being considered
  constexpr unsigned int val = ((X >> (Axis - 1)) & 1);

  // If there are more bits to consider, call the function recursively
  // with `Axis` decremented by 1. Otherwise, return the value of the current
  // bit.
  if constexpr (Axis > 1)
    return val + recursiveBitSum<X, Axis - 1>();
  else
    return val;
}

/**
 * Converts a static array of boolean values to an unsigned integer mask.
 * The parameter `BitArray` is a reference to the static array, and `N` is the
 * size of the array. The optional parameter `Axis` specifies the number of
 * elements in the array to consider, with `Axis` equal to `N` giving the full
 * mask. The function can be called with different values of `Axis` to create a
 * mask from a subset of the elements in the array. The function returns the
 * mask as an unsigned integer value.
 *
 * Example usage:
 *  With:
 *    constexpr bool bitArray[4] = { true, false, true, false };
 *  Convert the array to the unsigned integer mask:
 *    bitArrayToMask<4, bitArray>() // 1010 (decimal 10)
 *  Convert the first 3 elements of the array to the unsigned integer mask:
 *    bitArrayToMask<4, bitArray, 3>() // 100 (decimal 4)
 */
template <size_t N, const bool (&BitArray)[N], int Axis = N>
inline constexpr unsigned int bitArrayToMask() {
  // Calculate the value of the current element being considered and shift it
  // left by the appropriate number of bits
  constexpr unsigned int mask = (1 << (N - Axis)) * BitArray[Axis - 1];

  // If there are more elements to consider, call the function recursively with
  // `Axis` decremented by 1. Otherwise, return the value of the current
  // element.
  if constexpr (Axis > 1) {
    return mask + bitArrayToMask<N, BitArray, Axis - 1>();
  } else {
    return (1 << (N - 1)) * BitArray[0];
  }
}

/**
 * Generates a mask of the lower `N` bits of an `int`.
 * The optional parameter `Index` specifies the index of the current bit being
 * considered, with `Index` equal to 0 giving the full mask. The function can be
 * called with different values of `Index` to create a mask from a subset of the
 * bits. The function returns the mask as an unsigned integer value.
 *
 * Example usage:
 *  Generate a mask of the lower 3 bits of an int:
 *    maskNLower<3>() // decimal 7, binary 111
 *  Generates a mask of the lower 3 bits of an int, considering only the second
 *  and third bits:
 *    maskNLower<3, 1>() // decimal 6, binary 110
 */
template <int N, int Index = 0> inline constexpr unsigned int maskNLower() {
  static_assert(N < sizeof(unsigned int) * 8);
  static_assert(Index <= N);

  constexpr unsigned int val = 1 << Index;
  if constexpr (Index < N - 1)
    return val | maskNLower<N, Index + 1>();
  else
    return val;
}

template <class NumericType, int D, unsigned int axisMask = -1U>
class psPointLocator {
public:
  // Calculate the number of dimensions used based on the provided axis mask
  static constexpr unsigned int numActiveDimensions =
      recursiveBitSum<axisMask, D>();

  // Ensure that at least one dimension is used
  static_assert(
      numActiveDimensions >= 1,
      "The provided axes exclude mask would mask out all axes, but at "
      "least one axis must remain unmask for the tree to work "
      "properly.");

  // Types used in the class
  using VectorType = std::array<NumericType, D>;
  using PointType = std::array<NumericType, numActiveDimensions>;
  using SizeType = std::size_t;

  // Pure virtual functions to be implemented by derived classes
  virtual void build() = 0;
  virtual void setPoints(std::vector<VectorType> &passedPoints) = 0;
  virtual void setScalingFactors(const PointType &passedScalingFactors) = 0;
  virtual std::pair<SizeType, NumericType> findNearest(const PointType &x) = 0;
  virtual psSmartPointer<std::vector<std::pair<SizeType, NumericType>>>
  findKNearest(const PointType &x, const int k) = 0;
  virtual psSmartPointer<std::vector<std::pair<SizeType, NumericType>>>
  findNearestWithinRadius(const PointType &x, const NumericType radius) = 0;

  // Helper function for checking whether a given axis is enabled by the
  // axisMask
  inline constexpr bool isActive(SizeType axis) const {
    return ((axisMask >> axis) & SizeType{1});
  }

  // Helper function for converting from VectorType to PointType
  inline constexpr PointType applyMask(const VectorType &vec) const {
    SizeType j = 0;
    PointType converted{};
    for (SizeType i = 0; i < D; ++i)
      if (isActive(i)) {
        converted[j] = vec[i];
        ++j;
      }
    return converted;
  }
};
#endif