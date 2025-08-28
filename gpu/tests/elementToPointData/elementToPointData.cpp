#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <vector>

#include <vcTestAsserts.hpp>

// Simplified test implementation based on the actual ElementToPointData
// functions
namespace test {

template <typename NumericType> struct MinMaxInfo {
  NumericType min1, min2, max1, max2;
  int minIdx1, minIdx2, maxIdx1, maxIdx2;

  MinMaxInfo()
      : min1(std::numeric_limits<NumericType>::max()),
        min2(std::numeric_limits<NumericType>::max()), max1(0), max2(0),
        minIdx1(-1), minIdx2(-1), maxIdx1(-1), maxIdx2(-1) {}
};

template <typename NumericType, typename MeshNT>
MinMaxInfo<NumericType> findMinMaxValues(
    const std::vector<NumericType> &weights,
    const std::vector<std::pair<std::size_t, NumericType>> &closePoints,
    const std::vector<MeshNT> &elementData, unsigned j, unsigned numElements) {
  MinMaxInfo<NumericType> info;

  for (std::size_t k = 0; k < closePoints.size(); ++k) {
    if (weights[k] != NumericType(0)) {
      const unsigned elementIdx = closePoints[k].first + j * numElements;
      const auto value = elementData[elementIdx];

      // Update min values
      if (value < info.min1) {
        info.min2 = info.min1;
        info.minIdx2 = info.minIdx1;
        info.min1 = value;
        info.minIdx1 = static_cast<int>(k);
      } else if (value < info.min2) {
        info.min2 = value;
        info.minIdx2 = static_cast<int>(k);
      }

      // Update max values
      if (value > info.max1) {
        info.max2 = info.max1;
        info.maxIdx2 = info.maxIdx1;
        info.max1 = value;
        info.maxIdx1 = static_cast<int>(k);
      } else if (value > info.max2) {
        info.max2 = value;
        info.maxIdx2 = static_cast<int>(k);
      }
    }
  }

  return info;
}

template <typename NumericType, typename MeshNT>
void discardOutliers(
    std::vector<NumericType> &weightsCopy,
    const std::vector<NumericType> &weights,
    const std::vector<std::pair<std::size_t, NumericType>> &closePoints,
    const std::vector<MeshNT> &elementData, unsigned j, unsigned numElements,
    unsigned numClosePoints) {

  static constexpr bool discard2 = true;
  static constexpr bool discard4 = true;

  if (discard4 && numClosePoints > 4) {
    const auto info =
        findMinMaxValues(weights, closePoints, elementData, j, numElements);

    if (info.maxIdx1 != -1 && info.maxIdx2 != -1 && info.minIdx1 != -1 &&
        info.minIdx2 != -1) {
      weightsCopy[info.minIdx1] = NumericType(0);
      weightsCopy[info.minIdx2] = NumericType(0);
      weightsCopy[info.maxIdx1] = NumericType(0);
      weightsCopy[info.maxIdx2] = NumericType(0);
    }
  } else if (discard2 && numClosePoints > 2) {
    const auto info =
        findMinMaxValues(weights, closePoints, elementData, j, numElements);

    if (info.minIdx1 != -1 && info.maxIdx1 != -1) {
      weightsCopy[info.minIdx1] = NumericType(0);
      weightsCopy[info.maxIdx1] = NumericType(0);
    }
  }
}

} // namespace test

void testFindMinMaxValues() {
  // Test data
  std::vector<double> weights = {1.0, 0.0, 0.5, 0.8, 0.3, 1.2, 0.0, 0.9};
  std::vector<std::pair<std::size_t, double>> closePoints;
  for (size_t i = 0; i < weights.size(); ++i) {
    closePoints.emplace_back(i, 0.0); // Distance doesn't matter for this test
  }

  // Element data: values are [10.0, 0.0, 5.0, 15.0, 3.0, 20.0, 0.0, 8.0]
  std::vector<double> elementData = {10.0, 0.0, 5.0, 15.0, 3.0, 20.0, 0.0, 8.0};
  unsigned j = 0; // Data index
  unsigned numElements = elementData.size();

  auto result =
      test::findMinMaxValues(weights, closePoints, elementData, j, numElements);

  // Expected results:
  // Valid weights are at indices: 0, 2, 3, 4, 5, 7 (weights > 0)
  // Corresponding values: 10.0, 5.0, 15.0, 3.0, 20.0, 8.0
  // min1 = 3.0 (index 4), min2 = 5.0 (index 2)
  // max1 = 20.0 (index 5), max2 = 15.0 (index 3)

  std::cout << "Min1: " << result.min1 << " (idx: " << result.minIdx1 << ")"
            << std::endl;
  std::cout << "Min2: " << result.min2 << " (idx: " << result.minIdx2 << ")"
            << std::endl;
  std::cout << "Max1: " << result.max1 << " (idx: " << result.maxIdx1 << ")"
            << std::endl;
  std::cout << "Max2: " << result.max2 << " (idx: " << result.maxIdx2 << ")"
            << std::endl;

  // Verify results
  VC_TEST_ASSERT(result.min1 == 3.0);
  VC_TEST_ASSERT(result.minIdx1 == 4);
  VC_TEST_ASSERT(result.min2 == 5.0);
  VC_TEST_ASSERT(result.minIdx2 == 2);
  VC_TEST_ASSERT(result.max1 == 20.0);
  VC_TEST_ASSERT(result.maxIdx1 == 5);
  VC_TEST_ASSERT(result.max2 == 15.0);
  VC_TEST_ASSERT(result.maxIdx2 == 3);
}

void testFindMinMaxValuesEmptyWeights() {
  // Test with all zero weights
  std::vector<double> weights = {0.0, 0.0, 0.0, 0.0};
  std::vector<std::pair<std::size_t, double>> closePoints;
  for (size_t i = 0; i < weights.size(); ++i) {
    closePoints.emplace_back(i, 0.0);
  }

  std::vector<double> elementData = {10.0, 5.0, 15.0, 3.0};
  unsigned j = 0;
  unsigned numElements = elementData.size();

  auto result =
      test::findMinMaxValues(weights, closePoints, elementData, j, numElements);

  // All indices should remain -1 since no valid weights
  VC_TEST_ASSERT(result.minIdx1 == -1);
  VC_TEST_ASSERT(result.minIdx2 == -1);
  VC_TEST_ASSERT(result.maxIdx1 == -1);
  VC_TEST_ASSERT(result.maxIdx2 == -1);
}

void testDiscardOutliersDiscard4() {
  // Test data with 6 points (> 4, so discard4 should activate)
  std::vector<double> weights = {1.0, 0.5, 0.8, 0.3, 1.2, 0.9};
  std::vector<double> weightsCopy = weights;
  std::vector<std::pair<std::size_t, double>> closePoints;
  for (size_t i = 0; i < weights.size(); ++i) {
    closePoints.emplace_back(i, 0.0);
  }

  // Element data: values are [10.0, 5.0, 15.0, 3.0, 20.0, 8.0]
  // min1 = 3.0 (idx 3), min2 = 5.0 (idx 1)
  // max1 = 20.0 (idx 4), max2 = 15.0 (idx 2)
  std::vector<double> elementData = {10.0, 5.0, 15.0, 3.0, 20.0, 8.0};
  unsigned j = 0;
  unsigned numElements = elementData.size();
  unsigned numClosePoints = 6;

  test::discardOutliers(weightsCopy, weights, closePoints, elementData, j,
                        numElements, numClosePoints);

  // Check that the outliers were discarded (weights set to 0)
  // Indices 1, 2, 3, 4 should be zeroed (min1, min2, max1, max2)
  std::cout << "Weights after discarding outliers: ";
  for (size_t i = 0; i < weightsCopy.size(); ++i) {
    std::cout << weightsCopy[i] << " ";
  }
  std::cout << std::endl;

  VC_TEST_ASSERT(weightsCopy[1] == 0.0); // min2
  VC_TEST_ASSERT(weightsCopy[2] == 0.0); // max2
  VC_TEST_ASSERT(weightsCopy[3] == 0.0); // min1
  VC_TEST_ASSERT(weightsCopy[4] == 0.0); // max1
  // Indices 0 and 5 should remain unchanged
  VC_TEST_ASSERT(weightsCopy[0] == 1.0);
  VC_TEST_ASSERT(weightsCopy[5] == 0.9);
}

void testDiscardOutliersDiscard2() {
  // Test data with 3 points (> 2 but <= 4, so discard2 should activate)
  std::vector<double> weights = {1.0, 0.5, 0.8};
  std::vector<double> weightsCopy = weights;
  std::vector<std::pair<std::size_t, double>> closePoints;
  for (size_t i = 0; i < weights.size(); ++i) {
    closePoints.emplace_back(i, 0.0);
  }

  // Element data: values are [10.0, 3.0, 15.0]
  // min1 = 3.0 (idx 1), max1 = 15.0 (idx 2)
  std::vector<double> elementData = {10.0, 3.0, 15.0};
  unsigned j = 0;
  unsigned numElements = elementData.size();
  unsigned numClosePoints = 3;

  test::discardOutliers(weightsCopy, weights, closePoints, elementData, j,
                        numElements, numClosePoints);

  // Check that min1 and max1 were discarded
  std::cout << "Weights after discarding outliers (discard2): ";
  for (size_t i = 0; i < weightsCopy.size(); ++i) {
    std::cout << weightsCopy[i] << " ";
  }
  std::cout << std::endl;

  VC_TEST_ASSERT(weightsCopy[1] == 0.0); // min1
  VC_TEST_ASSERT(weightsCopy[2] == 0.0); // max1
  VC_TEST_ASSERT(weightsCopy[0] == 1.0); // unchanged
}

void testDiscardOutliersNoDiscard() {
  // Test data with 2 points (<= 2, so no discarding should happen)
  std::vector<double> weights = {1.0, 0.5};
  std::vector<double> weightsCopy = weights;
  std::vector<std::pair<std::size_t, double>> closePoints;
  for (size_t i = 0; i < weights.size(); ++i) {
    closePoints.emplace_back(i, 0.0);
  }

  std::vector<double> elementData = {10.0, 3.0};
  unsigned j = 0;
  unsigned numElements = elementData.size();
  unsigned numClosePoints = 2;

  test::discardOutliers(weightsCopy, weights, closePoints, elementData, j,
                        numElements, numClosePoints);

  // No weights should be changed
  for (size_t i = 0; i < weightsCopy.size(); ++i) {
    VC_TEST_ASSERT(weightsCopy[i] == weights[i]);
  }
}

void testFindMinMaxValuesSingleValue() {
  // Test with only one valid weight
  std::vector<double> weights = {0.0, 1.0, 0.0};
  std::vector<std::pair<std::size_t, double>> closePoints;
  for (size_t i = 0; i < weights.size(); ++i) {
    closePoints.emplace_back(i, 0.0);
  }

  std::vector<double> elementData = {10.0, 5.0, 15.0};
  unsigned j = 0;
  unsigned numElements = elementData.size();

  auto result =
      test::findMinMaxValues(weights, closePoints, elementData, j, numElements);

  // With only one value, min1 and max1 should be the same
  VC_TEST_ASSERT(result.min1 == 5.0);
  VC_TEST_ASSERT(result.minIdx1 == 1);
  VC_TEST_ASSERT(result.max1 == 5.0);
  VC_TEST_ASSERT(result.maxIdx1 == 1);
  // min2 and max2 should remain unset
  VC_TEST_ASSERT(result.minIdx2 == -1);
  VC_TEST_ASSERT(result.maxIdx2 == -1);
}

int main() {
  testFindMinMaxValues();
  testFindMinMaxValuesEmptyWeights();
  testFindMinMaxValuesSingleValue();
  testDiscardOutliersDiscard4();
  testDiscardOutliersDiscard2();
  testDiscardOutliersNoDiscard();
}
