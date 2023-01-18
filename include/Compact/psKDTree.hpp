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
#include <array>
#include <memory>
#include <optional>
#include <vector>

#include <lsMessage.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <psPointLocator.hpp>
#include <psQueues.hpp>
#include <psSmartPointer.hpp>

template <class NumericType> class psKDTree : psPointLocator<NumericType> {
  using Parent = psPointLocator<NumericType>;

  using typename Parent::PointType;
  using typename Parent::SizeType;

  class Node {
    PointType value{};
    SizeType index{};

  public:
    int axis{};
    Node *left = nullptr;
    Node *right = nullptr;

    Node(const PointType &passedValue, SizeType passedIndex)
        : value(passedValue), index(passedIndex) {}

    Node(Node &&other) {
      value.swap(other.value);
      index = other.index;
      axis = other.axis;

      left = other.left;
      right = other.right;

      other.left = nullptr;
      other.right = nullptr;
    }

    Node &operator=(Node &&other) {
      value.swap(other.value);
      index = other.index;
      axis = other.axis;

      left = other.left;
      right = other.right;

      other.left = nullptr;
      other.right = nullptr;

      return *this;
    }

    const PointType &getValue() const { return value; }
    SizeType getIndex() const { return index; }
  };

  template <class Iterator>
  static typename Iterator::pointer toRawPointer(const Iterator it) {
    return &(*it);
  }

  PointType Diff(const PointType &pVecA, const PointType &pVecB) const {
    PointType diff(pVecA.size(), 0.);
    for (SizeType i = 0; i < D; ++i)
      diff[i] = scalingFactors[i] * (pVecA[i] - pVecB[i]);
    return diff;
  }

  NumericType SquaredDistance(const PointType &pVecA,
                              const PointType &pVecB) const {
    auto diff = Diff(pVecA, pVecB);
    NumericType norm = 0;
    std::for_each(diff.begin(), diff.end(),
                  [&norm](NumericType entry) { norm += entry * entry; });
    return norm;
  }

  NumericType Distance(const PointType &pVecA, const PointType &pVecB) const {
    return std::sqrt(SquaredDistance(pVecA, pVecB));
  }

  void build(Node *parent, typename std::vector<Node>::iterator start,
             typename std::vector<Node>::iterator end, int depth, bool isLeft,
             int surplusWorkers, int maxParallelDepth) const {
    SizeType size = end - start;

    int axis = depth % D;

    if (size > 1) {
      SizeType medianIndex = (size + 1) / 2 - 1;
      std::nth_element(start, start + medianIndex, end,
                       [axis](Node &a, Node &b) {
                         return a.getValue()[axis] < b.getValue()[axis];
                       });

      Node *current = toRawPointer(start + medianIndex);
      current->axis = axis;

      if (isLeft)
        parent->left = current;
      else
        parent->right = current;

#ifdef _OPENMP
      bool dontSpawnMoreThreads = depth > maxParallelDepth + 1 ||
                                  (depth == maxParallelDepth + 1 &&
                                   omp_get_thread_num() >= surplusWorkers);
#endif
#pragma omp task final(dontSpawnMoreThreads)
      {
        // Left Subtree
        build(current,             // Use current node as parent
              start,               // Data start
              start + medianIndex, // Data end
              depth + 1,           // Depth
              true,                // Left
              surplusWorkers, maxParallelDepth);
      }

      //  Right Subtree
      build(current,                 // Use current node as parent
            start + medianIndex + 1, // Data start
            end,                     // Data end
            depth + 1,               // Depth
            false,                   // Right
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

private:
  /****************************************************************************
   * Recursive Tree Traversal                                                 *
   ****************************************************************************/
  void traverseDown(Node *currentNode, std::pair<NumericType, Node *> &best,
                    const PointType &x) const {
    if (currentNode == nullptr)
      return;

    int axis = currentNode->axis;

    // For distance comparison operations we only use the "reduced" aka less
    // compute intensive, but order preserving version of the distance
    // function.
    NumericType distance = SquaredDistance(x, currentNode->getValue());
    if (distance < best.first)
      best = std::pair{distance, currentNode};

    bool isLeft;
    if (x[axis] < currentNode->getValue()[axis]) {
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
    NumericType distanceToHyperplane =
        scalingFactors[axis] *
        std::abs(x[axis] - currentNode->getValue()[axis]);
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
                std::is_same_v<Q, cmBoundedPQueue<NumericType, Node *>> ||
                std::is_same_v<Q, cmClampedPQueue<NumericType, Node *>>>>
  void traverseDown(Node *currentNode, Q &queue, const PointType &x) const {
    if (currentNode == nullptr)
      return;

    int axis = currentNode->axis;

    // For distance comparison operations we only use the "reduced" aka less
    // compute intensive, but order preserving version of the distance
    // function.
    queue.enqueue(
        std::pair{SquaredDistance(x, currentNode->getValue()), currentNode});

    bool isLeft;
    if (x[axis] < currentNode->getValue()[axis]) {
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
    NumericType distanceToHyperplane =
        scalingFactors[axis] *
        std::abs(x[axis] - currentNode->getValue()[axis]);
    distanceToHyperplane *= distanceToHyperplane;

    bool intersects = false;
    if constexpr (std::is_same_v<Q, cmBoundedPQueue<NumericType, Node *>>) {
      intersects = queue.size() < queue.maxSize() ||
                   distanceToHyperplane < queue.worst();
    } else if constexpr (std::is_same_v<Q,
                                        cmClampedPQueue<NumericType, Node *>>) {
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

  static constexpr int intLog2(int x) {
    int val = 0;
    while (x >>= 1)
      ++val;
    return val;
  }

public:
  psKDTree() {}

  psKDTree(const std::vector<PointType> &passedPoints) {
    // The first row determins the data dimension
    D = passedPoints.at(0).size();

    // Initialize the scaling factors to one
    scalingFactors = PointType(D, 1.);

    // Create a vector of nodes
    nodes.reserve(passedPoints.size());
    {
      for (SizeType i = 0; i < passedPoints.size(); ++i) {
        nodes.emplace_back(Node{passedPoints[i], i});
      }
    }
  }

  void setPoints(const std::vector<PointType> &passedPoints) override {
    // The first row determins the data dimension
    D = passedPoints.at(0).size();

    // Initialize the scaling factors to one
    scalingFactors = PointType(D, 1.);

    nodes.reserve(passedPoints.size());
    {
      for (SizeType i = 0; i < passedPoints.size(); ++i) {
        nodes.emplace_back(Node{passedPoints[i], i});
      }
    }
  }

  void setScalingFactors(const PointType &passedScalingFactors) override {
    scalingFactors.clear();
    std::copy(passedScalingFactors.begin(), passedScalingFactors.end(),
              std::back_inserter(scalingFactors));
  }

  void build() override {
    if (nodes.size() == 0) {
      lsMessage::getInstance().addWarning("No points provided!").print();
      return;
    }

    // Local variable definitions of class member variables. These are needed
    // for the omp sharing costruct to work under MSVC
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

        SizeType size = myNodes.end() - myNodes.begin();
        SizeType medianIndex = (size + 1) / 2 - 1;

        std::nth_element(
            myNodes.begin(), myNodes.begin() + medianIndex, myNodes.end(),
            [](Node &a, Node &b) { return a.getValue()[0] < b.getValue()[0]; });

        myRootNode = &myNodes[medianIndex];
        myRootNode->axis = 0;

#ifdef _OPENMP
        bool dontSpawnMoreThreads = 0 > maxParallelDepth + 1 ||
                                    (0 == maxParallelDepth + 1 &&
                                     omp_get_thread_num() >= surplusWorkers);
#endif
#pragma omp task final(dontSpawnMoreThreads)
        {
          // Left Subtree
          build(myRootNode,                    // Use rootNode as parent
                myNodes.begin(),               // Data start
                myNodes.begin() + medianIndex, // Data end
                1,                             // Depth
                true,                          // Left
                surplusWorkers, maxParallelDepth);
        }

        // Right Subtree
        build(myRootNode,                        // Use rootNode as parent
              myNodes.begin() + medianIndex + 1, // Data start
              myNodes.end(),                     // Data end
              1,                                 // Depth
              false,                             // Right
              surplusWorkers, maxParallelDepth);
#pragma omp taskwait
      }
    }

    rootNode = myRootNode;
  }

  std::optional<std::pair<SizeType, NumericType>>
  findNearest(const PointType &x) const override {
    if (!rootNode)
      return {};

    auto best =
        std::pair{std::numeric_limits<NumericType>::infinity(), rootNode};
    traverseDown(rootNode, best, x);
    return std::pair{best.second->getIndex(),
                     Distance(x, best.second->getValue())};
  }

  std::optional<std::vector<std::pair<SizeType, NumericType>>>
  findKNearest(const PointType &x, const int k) const override {
    if (!rootNode)
      return {};

    auto queue = cmBoundedPQueue<NumericType, Node *>(k);
    traverseDown(rootNode, queue, x);

    auto result = std::vector<std::pair<SizeType, NumericType>>();
    result.reserve(k);

    while (!queue.empty()) {
      auto best = queue.dequeueBest();
      result.emplace_back(
          std::pair{best->getIndex(), Distance(x, best->getValue())});
    }
    return result;
  }

  std::optional<std::vector<std::pair<SizeType, NumericType>>>
  findNearestWithinRadius(const PointType &x,
                          const NumericType radius) const override {
    if (!rootNode)
      return {};

    auto queue = cmClampedPQueue<NumericType, Node *>(radius);
    traverseDown(rootNode, queue, x);

    auto result = std::vector<std::pair<SizeType, NumericType>>();
    result.reserve(queue.size());

    while (!queue.empty()) {
      auto best = queue.dequeueBest();
      result.emplace_back(
          std::pair{best->getIndex(), Distance(x, best->getValue())});
    }
    return result;
  }

private:
  SizeType D = 0;
  PointType scalingFactors;
  std::vector<Node> nodes;
  Node *rootNode = nullptr;
};

#endif