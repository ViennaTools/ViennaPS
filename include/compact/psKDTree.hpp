#ifndef PS_KDTREE_HPP
#define PS_KDTREE_HPP

// Inspired by the implementation of a parallelized kD-Tree by Francesco
// Andreuzzi (https://github.com/fAndreuzzi/parallel-kd-tree)
//
// --------------------- BEGIN ORIGINAL COPYRIGHT NOTICE ---------------------//
// MIT License
//
// Copyright (c) 2021 Francesco Andreuzzi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ---------------------- END ORIGINAL COPYRIGHT NOTICE ----------------------//

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <psLogger.hpp>
#include <psQueues.hpp>

template <class NumericType, class ValueType = std::vector<NumericType>>
class psKDTree {
  typedef typename std::vector<NumericType>::size_type SizeType;

  struct Node;

  SizeType D = 0;
  std::vector<NumericType> scalingFactors;
  std::vector<Node> nodes;

  Node *rootNode = nullptr;

public:
  psKDTree() {}

  psKDTree(const std::vector<ValueType> &passedPoints) {
    if (!passedPoints.empty()) {
      // The first row determins the data dimension
      D = passedPoints[0].size();

      // Initialize the scaling factors to one
      scalingFactors = std::vector<NumericType>(D, 1.);

      // Create a vector of nodes
      nodes.reserve(passedPoints.size());
      {
        for (SizeType i = 0; i < passedPoints.size(); ++i) {
          nodes.emplace_back(passedPoints[i], i);
        }
      }
    } else {
      psLogger::getInstance()
          .addWarning("psKDTree: the provided points vector is empty.")
          .print();
      return;
    }
  }

  void setPoints(const std::vector<ValueType> &passedPoints,
                 const std::vector<NumericType> &passedScalingFactors = {}) {
    if (passedPoints.empty()) {
      psLogger::getInstance()
          .addWarning("psKDTree: the provided points vector is empty.")
          .print();
      return;
    }

    // The first row determins the data dimension
    D = passedPoints[0].size();

    scalingFactors.clear();
    if (passedScalingFactors.empty()) {
      // Initialize the scaling factors to one
      scalingFactors = std::vector<NumericType>(D, 1.);
    } else {
      assert(
          passedScalingFactors.size() == D &&
          "The provided scaling factors have a different dimensionality than "
          "the data.");

      std::copy(passedScalingFactors.begin(), passedScalingFactors.end(),
                std::back_inserter(scalingFactors));
    }

    nodes.clear();
    nodes.reserve(passedPoints.size());
    for (SizeType i = 0; i < passedPoints.size(); ++i) {
      nodes.emplace_back(passedPoints[i], i);
    }
  }

  [[nodiscard]] std::optional<std::pair<SizeType, NumericType>>
  findNearest(const ValueType &x) const {
    if (!rootNode)
      return {};

    auto best =
        std::pair{std::numeric_limits<NumericType>::infinity(), rootNode};
    traverseDown(rootNode, best, x);
    return std::pair{best.second->index, Distance(x, best.second->value)};
  }

  [[nodiscard]] std::optional<std::vector<std::pair<SizeType, NumericType>>>
  findKNearest(const ValueType &x, const int k) const {
    if (!rootNode)
      return {};

    auto queue = psBoundedPQueue<NumericType, Node *>(k);
    traverseDown(rootNode, queue, x);

    auto result = std::vector<std::pair<SizeType, NumericType>>();
    result.reserve(k);

    while (!queue.empty()) {
      auto best = queue.dequeueBest();
      result.emplace_back(best->index, Distance(x, best->value));
    }
    return result;
  }

  [[nodiscard]] std::optional<std::vector<std::pair<SizeType, NumericType>>>
  findNearestWithinRadius(const ValueType &x, const NumericType radius) const {
    if (!rootNode)
      return {};

    auto queue = psClampedPQueue<NumericType, Node *>(radius);
    traverseDown(rootNode, queue, x);

    auto result = std::vector<std::pair<SizeType, NumericType>>();
    result.reserve(queue.size());

    while (!queue.empty()) {
      auto best = queue.dequeueBest();
      result.emplace_back(best->index, Distance(x, best->value));
    }
    return result;
  }

  void build() {
    if (nodes.size() == 0) {
      psLogger::getInstance().addWarning("KDTree: No points provided!").print();
      return;
    }

    // Local variable definitions of class member variables. These are needed
    // for the omp sharing construct to work under MSVC
    Node *myRootNode = nullptr;
    std::vector<Node> &myNodes = nodes;

#pragma omp parallel default(none) shared(myNodes, myRootNode)
    {
      int numThreads = 1;
#pragma omp single
      {
#ifdef _OPENMP
        numThreads = omp_get_num_threads();
#endif
        int maxParallelDepth = intLog2(numThreads);
        int surplusWorkers = numThreads - (1 << maxParallelDepth);

        auto size = static_cast<int>(myNodes.size());
        auto medianIndex = (size + 1) / 2 - 1;

        std::nth_element(
            myNodes.begin(), std::next(myNodes.begin(), medianIndex),
            myNodes.end(),
            [](Node &a, Node &b) { return a.value[0] < b.value[0]; });

        myRootNode = &myNodes[static_cast<SizeType>(medianIndex)];
        myRootNode->axis = 0;

#ifdef _OPENMP
        bool dontSpawnMoreThreads = 0 > maxParallelDepth + 1 ||
                                    (0 == maxParallelDepth + 1 &&
                                     omp_get_thread_num() >= surplusWorkers);
#endif
#pragma omp task final(dontSpawnMoreThreads)
        {
          // Left Subtree
          build(myRootNode,      // Use rootNode as parent
                myNodes.begin(), // Data start
                std::next(myNodes.begin(), medianIndex), // Data end
                1,                                       // Depth
                true,                                    // Left
                surplusWorkers, maxParallelDepth);
        }

        // Right Subtree
        build(myRootNode, // Use rootNode as parent
              std::next(myNodes.begin(), medianIndex + 1), // Data start
              myNodes.end(),                               // Data end
              1,                                           // Depth
              false,                                       // Right
              surplusWorkers, maxParallelDepth);
#pragma omp taskwait
      }
    }

    rootNode = myRootNode;
  }

private:
  void build(Node *parent, typename std::vector<Node>::iterator start,
             typename std::vector<Node>::iterator end, SizeType depth,
             bool isLeft, int surplusWorkers, int maxParallelDepth) const {
    auto size = std::distance(start, end);
    auto axis = depth % D;

    if (size > 1) {
      auto medianIndex = (size + 1) / 2 - 1;
      std::nth_element(
          start, std::next(start, medianIndex), end,
          [axis](Node &a, Node &b) { return a.value[axis] < b.value[axis]; });

      Node *current = toRawPointer(std::next(start, medianIndex));
      current->axis = axis;

      if (isLeft)
        parent->left = current;
      else
        parent->right = current;

#ifdef _OPENMP
      bool dontSpawnMoreThreads =
          static_cast<int>(depth) > maxParallelDepth + 1 ||
          (static_cast<int>(depth) == maxParallelDepth + 1 &&
           omp_get_thread_num() >= surplusWorkers);
#endif
#pragma omp task final(dontSpawnMoreThreads)
      {
        // Left Subtree
        build(current,                       // Use current node as parent
              start,                         // Data start
              std::next(start, medianIndex), // Data end
              depth + 1,                     // Depth
              true,                          // Left
              surplusWorkers, maxParallelDepth);
      }

      //  Right Subtree
      build(current,                           // Use current node as parent
            std::next(start, medianIndex + 1), // Data start
            end,                               // Data end
            depth + 1,                         // Depth
            false,                             // Right
            surplusWorkers, maxParallelDepth);
#pragma omp taskwait
    } else if (size == 1) {
      Node *current = toRawPointer(start);
      current->axis = axis;
      // Leaf Node
      if (isLeft)
        parent->left = current;
      else
        parent->right = current;
    }
  }

  /****************************************************************************
   * Recursive Tree Traversal                                                 *
   ****************************************************************************/
  void traverseDown(Node *currentNode, std::pair<NumericType, Node *> &best,
                    const ValueType &x) const {
    if (currentNode == nullptr)
      return;

    auto axis = currentNode->axis;

    // For distance comparison operations we only use the "reduced" aka less
    // compute intensive, but order preserving version of the distance
    // function.
    auto distance = SquaredDistance(x, currentNode->value);
    if (distance < best.first)
      best = std::pair{distance, currentNode};

    bool isLeft;
    if (x[axis] < currentNode->value[axis]) {
      traverseDown(currentNode->left, best, x);
      isLeft = true;
    } else {
      traverseDown(currentNode->right, best, x);
      isLeft = false;
    }

    // If the hypersphere with origin at x and a radius of our current best
    // distance intersects the hyperplane defined by the partitioning of the
    // current node, we also have to search the other subtree, since there could
    // be points closer to x than our current best.
    auto distanceToHyperplane =
        scalingFactors[axis] * std::abs(x[axis] - currentNode->value[axis]);
    distanceToHyperplane *= distanceToHyperplane;
    if (distanceToHyperplane < best.first) {
      if (isLeft)
        traverseDown(currentNode->right, best, x);
      else
        traverseDown(currentNode->left, best, x);
    }
    return;
  }

  template <typename Q,
            typename = std::enable_if_t<
                std::is_same_v<Q, psBoundedPQueue<NumericType, Node *>> ||
                std::is_same_v<Q, psClampedPQueue<NumericType, Node *>>>>
  void traverseDown(Node *currentNode, Q &queue,
                    const std::vector<NumericType> &x) const {
    if (currentNode == nullptr)
      return;

    int axis = currentNode->axis;

    // For distance comparison operations we only use the squared distance which
    // is less compute intensive, but order preserving version of the distance
    // function.
    queue.enqueue(
        std::pair{SquaredDistance(x, currentNode->value), currentNode});

    bool isLeft;
    if (x[axis] < currentNode->value[axis]) {
      traverseDown(currentNode->left, queue, x);
      isLeft = true;
    } else {
      traverseDown(currentNode->right, queue, x);
      isLeft = false;
    }

    // If the hypersphere with origin at x and a radius of our current best
    // distance intersects the hyperplane defined by the partitioning of the
    // current node, we also have to search the other subtree, since there could
    // be points closer to x than our current best.
    auto distanceToHyperplane =
        scalingFactors[axis] * std::abs(x[axis] - currentNode->value[axis]);
    distanceToHyperplane *= distanceToHyperplane;

    bool intersects = false;
    if constexpr (std::is_same_v<Q, psBoundedPQueue<NumericType, Node *>>) {
      intersects = queue.size() < queue.maxSize() ||
                   distanceToHyperplane < queue.worst();
    } else if constexpr (std::is_same_v<Q,
                                        psClampedPQueue<NumericType, Node *>>) {
      intersects = distanceToHyperplane < queue.worst();
    }

    if (intersects) {
      if (isLeft)
        traverseDown(currentNode->right, queue, x);
      else
        traverseDown(currentNode->left, queue, x);
    }
    return;
  }

  /****************************************************************************
   * Utility Functions                                                        *
   ****************************************************************************/

  // Converts Iterator to a raw pointer
  template <class Iterator>
  [[nodiscard]] static typename Iterator::pointer
  toRawPointer(const Iterator it) {
    return &(*it);
  }

  // Quickly calculate the log2 of signed ints
  template <typename SignedInt,
            typename = std::enable_if_t<std::is_integral_v<SignedInt> &&
                                        std::is_signed_v<SignedInt>>>
  [[nodiscard]] static constexpr SignedInt intLog2(SignedInt x) {
    SignedInt val = 0;
    while (x >>= 1)
      ++val;
    return val;
  }

  [[nodiscard]] NumericType SquaredDistance(const ValueType &pVecA,
                                            const ValueType &pVecB) const {
    NumericType norm = 0;
    for (SizeType i = 0; i < D; ++i)
      norm += scalingFactors[i] * scalingFactors[i] * (pVecA[i] - pVecB[i]) *
              (pVecA[i] - pVecB[i]);
    return norm;
  }

  [[nodiscard]] NumericType Distance(const ValueType &pVecA,
                                     const ValueType &pVecB) const {
    return std::sqrt(SquaredDistance(pVecA, pVecB));
  }

  /****************************************************************************
   * The Node struct implementation                                           *
   ****************************************************************************/
  struct Node {
    ValueType value{};
    SizeType index{};
    SizeType axis{};

    Node *left = nullptr;
    Node *right = nullptr;

    Node(const ValueType &passedValue, SizeType passedIndex) noexcept
        : value(passedValue), index(passedIndex) {}

    Node(Node &&other) noexcept {
      value.swap(other.value);
      index = other.index;
      axis = other.axis;

      left = other.left;
      right = other.right;

      other.left = nullptr;
      other.right = nullptr;
    }

    Node &operator=(Node &&other) noexcept {
      value.swap(other.value);
      index = other.index;
      axis = other.axis;

      left = other.left;
      right = other.right;

      other.left = nullptr;
      other.right = nullptr;

      return *this;
    }
  };
};

#endif