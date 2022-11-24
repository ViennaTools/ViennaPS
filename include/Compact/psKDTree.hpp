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
#include <vector>

#include <lsMessage.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <psPointLocator.hpp>
#include <psQueues.hpp>
#include <psSmartPointer.hpp>

template <class NumericType, int D, int Dim = D>
class psKDTree : psPointLocator<NumericType, D, Dim> {
  using typename psPointLocator<NumericType, D, Dim>::VectorType;
  using typename psPointLocator<NumericType, D, Dim>::PointType;
  using typename psPointLocator<NumericType, D, Dim>::SizeType;

  SizeType N;
  SizeType treeSize = 0;

  std::array<NumericType, D> scalingFactors{1.};

  NumericType gridDelta;

  struct Node {
    VectorType value;
    SizeType index;
    int axis;

    Node *left = nullptr;
    Node *right = nullptr;

    Node(VectorType &passedValue, SizeType passedIndex)
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
  };

  std::vector<Node> nodes;
  Node *rootNode = nullptr;

  template <class Iterator>
  static typename Iterator::pointer toRawPointer(const Iterator it) {
    return &(*it);
  }

  template <size_t N>
  std::array<NumericType, N>
  Diff(const std::array<NumericType, N> &pVecA,
       const std::array<NumericType, N> &pVecB) const {
    std::array<NumericType, N> diff{0};
    for (int i = 0; i < N; ++i)
      diff[i] = scalingFactors[i] * (pVecA[i] - pVecB[i]);
    return diff;
  }

  template <size_t N>
  NumericType Distance(const std::array<NumericType, N> &pVecA,
                       const std::array<NumericType, N> &pVecB) const {
    auto diff = Diff(pVecA, pVecB);
    NumericType norm = 0;
    std::for_each(diff.begin(), diff.end(),
                  [&norm](NumericType entry) { norm += entry * entry; });
    return std::sqrt(norm);
  }

  template <size_t M, size_t N>
  NumericType Distance(const std::array<NumericType, M> &pVecA,
                       const std::array<NumericType, N> &pVecB) const {
    constexpr SizeType S = std::min({M, N});

    std::vector<NumericType> diff(S, 0);
    for (int i = 0; i < S; ++i)
      diff[i] = scalingFactors[i] * (pVecA[i] - pVecB[i]);

    NumericType norm = 0;
    std::for_each(diff.begin(), diff.end(),
                  [&norm](NumericType entry) { norm += entry * entry; });
    return std::sqrt(norm);
  }

  void build(Node *parent, typename std::vector<Node>::iterator start,
             typename std::vector<Node>::iterator end, int depth, bool isLeft,
             int surplusWorkers, int maxParallelDepth) const {
    SizeType size = end - start;

    int axis = depth % D;

    if (size > 1) {
      SizeType medianIndex = (size + 1) / 2 - 1;
      std::nth_element(
          start, start + medianIndex, end,
          [axis](Node &a, Node &b) { return a.value[axis] < b.value[axis]; });

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
    NumericType distance = Distance(x, currentNode->value);
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
    NumericType distanceToHyperplane =
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
                std::is_same_v<Q, cmBoundedPQueue<NumericType, Node *>> ||
                std::is_same_v<Q, cmClampedPQueue<NumericType, Node *>>>>
  void traverseDown(Node *currentNode, Q &queue, const PointType &x) const {
    if (currentNode == nullptr)
      return;

    int axis = currentNode->axis;

    // For distance comparison operations we only use the "reduced" aka less
    // compute intensive, but order preserving version of the distance
    // function.
    queue.enqueue(std::pair{Distance(x, currentNode->value), currentNode});

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
    NumericType distanceToHyperplane =
        scalingFactors[axis] * std::abs(x[axis] - currentNode->value[axis]);
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
  psKDTree() {
    // Initialize the scaling factors to one
    for (auto &sf : scalingFactors)
      sf = 1.0;
  }

  psKDTree(std::vector<VectorType> &passedPoints) {
    // Initialize the scaling factors to one
    for (auto &sf : scalingFactors)
      sf = 1.0;

    // Create a vector of nodes
    nodes.reserve(passedPoints.size());
    {
      for (SizeType i = 0; i < passedPoints.size(); ++i) {
        nodes.emplace_back(Node{passedPoints[i], i});
      }
    }
  }

  void setPoints(std::vector<VectorType> &passedPoints) override {
    nodes.reserve(passedPoints.size());
    {
      for (SizeType i = 0; i < passedPoints.size(); ++i) {
        nodes.emplace_back(Node{passedPoints[i], i});
      }
    }
  }

  void setScalingFactors(
      const std::array<NumericType, D> &passedScalingFactors) override {
    scalingFactors = passedScalingFactors;
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
      int threadID = 0;
      int numThreads = 1;
#pragma omp single
      {
#ifdef _OPENMP
        threadID = omp_get_thread_num();
        numThreads = omp_get_num_threads();
#endif
        int maxParallelDepth = intLog2(numThreads);
        int surplusWorkers = numThreads - (1 << maxParallelDepth);

        SizeType size = myNodes.end() - myNodes.begin();
        SizeType medianIndex = (size + 1) / 2 - 1;

        std::nth_element(
            myNodes.begin(), myNodes.begin() + medianIndex, myNodes.end(),
            [](Node &a, Node &b) { return a.value[0] < b.value[0]; });

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

  std::pair<SizeType, NumericType>
  findNearest(const PointType &x) const override {
    auto best =
        std::pair{std::numeric_limits<NumericType>::infinity(), rootNode};
    traverseDown(rootNode, best, x);
    return {best.second->index, Distance(x, best.second->value)};
  }

  psSmartPointer<std::vector<std::pair<SizeType, NumericType>>>
  findKNearest(const PointType &x, const int k) const override {
    auto queue = cmBoundedPQueue<NumericType, Node *>(k);
    traverseDown(rootNode, queue, x);

    auto initial = std::pair<SizeType, NumericType>{
        rootNode->index, std::numeric_limits<NumericType>::infinity()};
    auto result =
        psSmartPointer<std::vector<std::pair<SizeType, NumericType>>>::New();

    while (!queue.empty()) {
      auto best = queue.dequeueBest();
      result->emplace_back(std::pair{best->index, Distance(x, best->value)});
    }
    return result;
  }

  psSmartPointer<std::vector<std::pair<SizeType, NumericType>>>
  findNearestWithinRadius(const PointType &x,
                          const NumericType radius) const override {
    auto queue = cmClampedPQueue<NumericType, Node *>(radius);
    traverseDown(rootNode, queue, x);

    auto initial = std::pair<SizeType, NumericType>{
        rootNode->index, std::numeric_limits<NumericType>::infinity()};
    auto result =
        psSmartPointer<std::vector<std::pair<SizeType, NumericType>>>::New();

    while (!queue.empty()) {
      auto best = queue.dequeueBest();
      result->emplace_back(std::pair{best->index, Distance(x, best->value)});
    }
    return result;
  }
};

#endif