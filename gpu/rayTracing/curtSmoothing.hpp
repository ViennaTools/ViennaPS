#pragma once

#include <rayUtil.hpp>

#include <lsMesh.hpp>

#include <psSmartPointer.hpp>

template <class NumericType, int D> class curtSmoothing {

  typedef std::vector<std::vector<unsigned int>> pointNeighborhoodType;

  psSmartPointer<lsMesh<NumericType>> pointCloud;
  psSmartPointer<std::vector<NumericType>> mData = nullptr;
  std::string mDataName;
  NumericType mSmoothingSize;
  pointNeighborhoodType mPointNeighborhood;

public:
  curtSmoothing() {}
  curtSmoothing(psSmartPointer<lsMesh<NumericType>> passedPointCloud)
      : pointCloud(passedPointCloud) {
    initPointNeighborhood();
  }
  curtSmoothing(psSmartPointer<lsMesh<NumericType>> passedPointCloud,
                const std::string passedDataName,
                const NumericType passedGridDelta)
      : pointCloud(passedPointCloud), mDataName(passedDataName),
        mSmoothingSize(passedGridDelta * 1.5) {
    initPointNeighborhood();
  }
  curtSmoothing(psSmartPointer<lsMesh<NumericType>> passedPointCloud,
                const psSmartPointer<std::vector<NumericType>> passedData,
                const NumericType passedGridDelta)
      : pointCloud(passedPointCloud), mData(passedData),
        mSmoothingSize(passedGridDelta * 1.5) {
    initPointNeighborhood();
  }

  void apply() {
    assert(mPointNeighborhood.size() > 0);

    if (!mData)
      mData = pointCloud->getCellData().getScalarData(mDataName);
    std::vector<NumericType> oldFlux(mData->begin(), mData->end());

#pragma omp parallel for
    for (size_t idx = 0; idx < mData->size(); idx++) {
      auto neighborhood = mPointNeighborhood[idx];
      for (auto const &nbi : neighborhood) {
        mData->at(idx) += oldFlux[nbi];
      }
      mData->at(idx) /= (neighborhood.size() + 1);
    }
  }

private:
  void initPointNeighborhood() {
    auto numPoints = pointCloud->nodes.size();
    auto &points = pointCloud->nodes;
    mPointNeighborhood.clear();
    mPointNeighborhood.resize(numPoints, std::vector<unsigned int>{});

    static_assert(D == 3);

    std::vector<unsigned int> side1;
    std::vector<unsigned int> side2;

    // create copy of bounding box
    std::array<NumericType, 3> min = pointCloud->minimumExtent;
    std::array<NumericType, 3> max = pointCloud->maximumExtent;

    std::vector<int> dirs;
    for (int i = 0; i < 3; ++i) {
      if (min[i] != max[i]) {
        dirs.push_back(i);
      }
    }
    dirs.shrink_to_fit();

    int dirIdx = 0;
    NumericType pivot = (max[dirs[dirIdx]] + min[dirs[dirIdx]]) / 2;

    // divide point data
    for (unsigned int idx = 0; idx < numPoints; ++idx) {
      if (points[idx][dirs[dirIdx]] <= pivot) {
        side1.push_back(idx);
      } else {
        side2.push_back(idx);
      }
    }
    createNeighborhood(points, side1, side2, min, max, dirIdx, dirs, pivot);
  }

  void createNeighborhood(const std::vector<std::array<NumericType, 3>> &points,
                          const std::vector<unsigned int> &side1,
                          const std::vector<unsigned int> &side2,
                          const std::array<NumericType, 3> &min,
                          const std::array<NumericType, 3> &max,
                          const int &dirIdx, const std::vector<int> &dirs,
                          const NumericType &pivot) {
    assert(0 <= dirIdx && dirIdx < dirs.size() && "Assumption");
    if (side1.size() + side2.size() <= 1) {
      return;
    }

    // Corner case
    // The pivot element should actually be inbetween min and max.
    if (pivot == min[dirs[dirIdx]] || pivot == max[dirs[dirIdx]]) {
      // In this case the points are extremly close to each other (with respect
      // to the floating point precision).
      assert((min[dirs[dirIdx]] + max[dirs[dirIdx]]) / 2 == pivot &&
             "Characterization of corner case");
      auto sides = std::vector<unsigned int>(side1);
      sides.insert(sides.end(), side2.begin(), side2.end());
      // Add each of them to the neighborhoods
      for (unsigned int idx1 = 0; idx1 < sides.size() - 1; ++idx1) {
        for (unsigned int idx2 = idx1 + 1; idx2 < sides.size(); ++idx2) {
          auto const &pi1 = sides[idx1];
          auto const &pi2 = sides[idx2];
          assert(pi1 != pi2 && "Assumption");
          mPointNeighborhood[pi1].push_back(pi2);
          mPointNeighborhood[pi2].push_back(pi1);
        }
      }
      return;
    }

    // sets of candidates
    std::vector<unsigned int> side1Cand;
    std::vector<unsigned int> side2Cand;

    int newDirIdx = (dirIdx + 1) % dirs.size();
    NumericType newPivot = (max[dirs[newDirIdx]] + min[dirs[newDirIdx]]) / 2;

    // recursion sets
    std::vector<unsigned int> s1r1set;
    std::vector<unsigned int> s1r2set;
    std::vector<unsigned int> s2r1set;
    std::vector<unsigned int> s2r2set;

    for (unsigned int idx = 0; idx < side1.size(); ++idx) {
      const auto &point = points[side1[idx]];
      assert(point[dirs[dirIdx]] <= pivot && "Correctness Assertion");
      if (point[dirs[newDirIdx]] <= newPivot) {
        s1r1set.push_back(side1[idx]);
      } else {
        s1r2set.push_back(side1[idx]);
      }
      if (point[dirs[dirIdx]] + mSmoothingSize <= pivot) {
        continue;
      }
      side1Cand.push_back(side1[idx]);
    }
    for (unsigned int idx = 0; idx < side2.size(); ++idx) {
      const auto &point = points[side2[idx]];
      assert(point[dirs[dirIdx]] > pivot && "Correctness Assertion");
      if (point[dirs[newDirIdx]] <= newPivot) {
        s2r1set.push_back(side2[idx]);
      } else {
        s2r2set.push_back(side2[idx]);
      }
      if (point[dirs[dirIdx]] - mSmoothingSize >= pivot) {
        continue;
      }
      side2Cand.push_back(side2[idx]);
    }

    // Iterate over pairs of candidates
    if (side1Cand.size() > 0 && side2Cand.size() > 0) {
      for (unsigned int ci1 = 0; ci1 < side1Cand.size(); ++ci1) {
        for (unsigned int ci2 = 0; ci2 < side2Cand.size(); ++ci2) {
          const auto &point1 = points[side1Cand[ci1]];
          const auto &point2 = points[side2Cand[ci2]];

          assert(std::abs(point1[dirs[dirIdx]] - point2[dirs[dirIdx]]) <=
                     (2 * mSmoothingSize) &&
                 "Correctness Assertion");
          if (checkDistance(point1, point2, mSmoothingSize)) {
            mPointNeighborhood[side1Cand[ci1]].push_back(side2Cand[ci2]);
            mPointNeighborhood[side2Cand[ci2]].push_back(side1Cand[ci1]);
          }
        }
      }
    }

    // Recurse
    if (side1.size() > 1) {
      auto newS1Max = max;
      newS1Max[dirs[dirIdx]] = pivot; // old diridx and old pivot!
      createNeighborhood(points, s1r1set, s1r2set, min, newS1Max, newDirIdx,
                         dirs, newPivot);
    }
    if (side2.size() > 1) {
      auto newS2Min = min;
      newS2Min[dirs[dirIdx]] = pivot; // old diridx and old pivot!
      createNeighborhood(points, s2r1set, s2r2set, newS2Min, max, newDirIdx,
                         dirs, newPivot);
    }
  }

  template <size_t Dim>
  bool checkDistance(const std::array<NumericType, Dim> &p1,
                     const std::array<NumericType, Dim> &p2,
                     const NumericType &dist) {
    for (int i = 0; i < D; ++i) {
      if (std::abs(p1[i] - p2[i]) >= dist)
        return false;
    }
    if (rayInternal::Distance<NumericType>(p1, p2) < dist)
      return true;

    return false;
  }
};