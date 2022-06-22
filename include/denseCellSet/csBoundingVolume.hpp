#pragma once

#include <array>
#include <set>

#include "csUtil.hpp"

template <class T> class csBoundingVolume {
private:
  using BVPtrType = lsSmartPointer<csBoundingVolume<T>>;
  using BoundsType = csPair<csTriple<T>>;
  using CellIdsPtr = std::set<unsigned> *;

  std::array<std::set<unsigned>, 8> cellIds;
  std::array<BoundsType, 8> bounds;
  std::array<BVPtrType, 8> links;
  BoundsType outBound;
  int layer = -1;

public:
  csBoundingVolume() {}
  csBoundingVolume(const BoundsType &outerBound, int thisLayer)
      : layer(thisLayer) {
    outBound = outerBound;
    buildBounds(outerBound);
    if (layer > 0) {
      for (size_t i = 0; i < 8; i++) {
        links[i] = BVPtrType::New(bounds[i], layer - 1);
      }
    }
  }

  CellIdsPtr getCellIds(const csTriple<T> point) {
    auto vid = getVolumeIndex(point);
    assert(vid < 8 && "Point in invalid BV");

    if (layer == 0) {
      return &cellIds[vid];
    }

    return links[vid]->getCellIds(point);
  }

  void getBoundingVolumeBounds(const csTriple<T> point) {
    auto vid = getVolumeIndex(point);
    assert(vid < 8 && "Point in invalid BV");

    if (layer == 0) {
      printBound(vid);
      return;
    }

    links[vid]->getBoundingVolumeBounds(point);
  }

  size_t getTotalCellCounts() {
    size_t count = 0;
    if (layer == 0) {
      for (size_t i = 0; i < 8; i++) {
        count += cellIds[i].size();
      }
      return count;
    } else {
      for (size_t i = 0; i < 8; i++) {
        count += links[i]->getTotalCellCounts();
      }
    }
    return count;
  }

  void clear() {
    if (layer == 0) {
      for (size_t i = 0; i < 8; i++) {
        cellIds[i].clear();
      }
    } else {
      for (size_t i = 0; i < 8; i++) {
        links[i]->clear();
      }
    }
  }

  BVPtrType getLink(const csTriple<T> point) {
    auto vid = getVolumeIndex(point);
    return getLink(vid);
  }

  BVPtrType getLink(size_t vid) { return links[vid]; }

  size_t getVolumeIndex(const csTriple<T> point) {
    size_t vid = 8;
    for (size_t idx = 0; idx < 8; idx++) {
      if (insideVolume(point, idx)) {
        vid = idx;
        break;
      }
    }
    return vid;
  }

  bool insideVolume(const csTriple<T> p, const size_t idx) const {
    return p[0] > bounds[idx][0][0] && p[0] <= bounds[idx][1][0] &&
           p[1] > bounds[idx][0][1] && p[1] <= bounds[idx][1][1] &&
           p[2] > bounds[idx][0][2] && p[2] <= bounds[idx][1][2];
  }

  bool insideVolume(const T x, const T y, const T z, const size_t idx) const {
    return x > bounds[idx][0][0] && x <= bounds[idx][1][0] &&
           y > bounds[idx][0][1] && y <= bounds[idx][1][1] &&
           z > bounds[idx][0][2] && z <= bounds[idx][1][2];
  }

  void printBound(size_t vid) {
    std::cout << "Bounding volume span: [" << bounds[vid][0][0] << ", "
              << bounds[vid][0][1] << ", " << bounds[vid][0][2] << "] - ["
              << bounds[vid][1][0] << ", " << bounds[vid][1][1] << ", "
              << bounds[vid][1][2] << "]\n";
  }

private:
  void buildBounds(const BoundsType &outerBound) {
    auto xExt = (outerBound[1][0] - outerBound[0][0]) / 2.;
    auto yExt = (outerBound[1][1] - outerBound[0][1]) / 2.;
    auto zExt = (outerBound[1][2] - outerBound[0][2]) / 2.;

    const auto BVH1 = std::array<csTriple<T>, 2>{
        outerBound[0][0],        outerBound[0][1],
        outerBound[0][2],        outerBound[0][0] + xExt,
        outerBound[0][1] + yExt, outerBound[0][2] + zExt};
    const auto BVH2 = std::array<csTriple<T>, 2>{
        outerBound[0][0] + xExt, outerBound[0][1],
        outerBound[0][2],        outerBound[0][0] + 2 * xExt,
        outerBound[0][1] + yExt, outerBound[0][2] + zExt};
    const auto BVH3 = std::array<csTriple<T>, 2>{outerBound[0][0] + xExt,
                                                 outerBound[0][1] + yExt,
                                                 outerBound[0][2],
                                                 outerBound[0][0] + 2 * xExt,
                                                 outerBound[0][1] + 2 * yExt,
                                                 outerBound[0][2] + zExt};
    const auto BVH4 = std::array<csTriple<T>, 2>{outerBound[0][0],
                                                 outerBound[0][1] + yExt,
                                                 outerBound[0][2],
                                                 outerBound[0][0] + xExt,
                                                 outerBound[0][1] + 2 * yExt,
                                                 outerBound[0][2] + zExt};

    // top
    const auto BVH5 = std::array<csTriple<T>, 2>{
        outerBound[0][0],        outerBound[0][1],
        outerBound[0][2] + zExt, outerBound[0][0] + xExt,
        outerBound[0][1] + yExt, outerBound[0][2] + 2 * zExt};
    const auto BVH6 = std::array<csTriple<T>, 2>{
        outerBound[0][0] + xExt, outerBound[0][1],
        outerBound[0][2] + zExt, outerBound[0][0] + 2 * xExt,
        outerBound[0][1] + yExt, outerBound[0][2] + 2 * zExt};
    const auto BVH7 = std::array<csTriple<T>, 2>{
        outerBound[0][0] + xExt,     outerBound[0][1] + yExt,
        outerBound[0][2] + zExt,     outerBound[0][0] + 2 * xExt,
        outerBound[0][1] + 2 * yExt, outerBound[0][2] + 2 * zExt};
    const auto BVH8 = std::array<csTriple<T>, 2>{outerBound[0][0],
                                                 outerBound[0][1] + yExt,
                                                 outerBound[0][2] + zExt,
                                                 outerBound[0][0] + xExt,
                                                 outerBound[0][1] + 2 * yExt,
                                                 outerBound[0][2] + 2 * zExt};
    bounds = std::array<std::array<csTriple<T>, 2>, 8>{BVH1, BVH2, BVH3, BVH4,
                                                       BVH5, BVH6, BVH7, BVH8};
  }
};
