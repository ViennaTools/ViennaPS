#pragma once

#include <array>
#include <set>

#include "csUtil.hpp"

template <class T, int D> class csBoundingVolume {
private:
  using BVPtrType = lsSmartPointer<csBoundingVolume<T, D>>;
  using BoundsType = csPair<std::array<T, D>>;
  using CellIdsPtr = std::set<unsigned> *;

  std::array<std::set<unsigned>, (1 << D)> cellIds;
  std::array<BoundsType, (1 << D)> bounds;
  std::array<BVPtrType, (1 << D)> links;
  BoundsType outBound;
  int layer = -1;

public:
  csBoundingVolume() {}
  csBoundingVolume(const BoundsType &outerBound, int thisLayer)
      : layer(thisLayer) {
    outBound = outerBound;
    if constexpr (D == 3)
      buildBounds3D(outerBound);
    else
      buildBounds2D(outerBound);
    if (layer > 0) {
      for (size_t i = 0; i < (1 << D); i++) {
        links[i] = BVPtrType::New(bounds[i], layer - 1);
      }
    }
  }

  CellIdsPtr getCellIds(const std::array<T, D> point) {
    auto vid = getVolumeIndex(point);
    assert(vid < (1 << D) && "Point in invalid BV");

    if (layer == 0) {
      return &cellIds[vid];
    }

    return links[vid]->getCellIds(point);
  }

  void getBoundingVolumeBounds(const std::array<T, D> point) {
    auto vid = getVolumeIndex(point);
    assert(vid < (1 << D) && "Point in invalid BV");

    if (layer == 0) {
      printBound(vid);
      return;
    }

    links[vid]->getBoundingVolumeBounds(point);
  }

  size_t getTotalCellCounts() {
    size_t count = 0;
    if (layer == 0) {
      for (size_t i = 0; i < (1 << D); i++) {
        count += cellIds[i].size();
      }
      return count;
    } else {
      for (size_t i = 0; i < (1 << D); i++) {
        count += links[i]->getTotalCellCounts();
      }
    }
    return count;
  }

  void clear() {
    if (layer == 0) {
      for (size_t i = 0; i < (1 << D); i++) {
        cellIds[i].clear();
      }
    } else {
      for (size_t i = 0; i < (1 << D); i++) {
        links[i]->clear();
      }
    }
  }

  BVPtrType getLink(const csTriple<T> point) {
    auto vid = getVolumeIndex(point);
    return getLink(vid);
  }

  BVPtrType getLink(size_t vid) { return links[vid]; }

  size_t getVolumeIndex(const std::array<T, D> point) {
    size_t vid = (1 << D);
    for (size_t idx = 0; idx < (1 << D); idx++) {
      if (insideVolume(point, idx)) {
        vid = idx;
        break;
      }
    }
    return vid;
  }

  bool insideVolume(const std::array<T, D> p, const size_t idx) const {
    if constexpr (D == 3) {
      return p[0] > bounds[idx][0][0] && p[0] <= bounds[idx][1][0] &&
             p[1] > bounds[idx][0][1] && p[1] <= bounds[idx][1][1] &&
             p[2] > bounds[idx][0][2] && p[2] <= bounds[idx][1][2];
    } else {
      return p[0] > bounds[idx][0][0] && p[0] <= bounds[idx][1][0] &&
             p[1] > bounds[idx][0][1] && p[1] <= bounds[idx][1][1];
    }
  }

  void printBound(size_t vid) {
    std::cout << "Bounding volume span: [";
    for (int i = 0; i < D; i++)
      std::cout << bounds[vid][0][i] << ", ";
    std::cout << "] - [";
    for (int i = 0; i < D; i++)
      std::cout << bounds[vid][1][i] << ", ";
    std::cout << "]\n";
  }

private:
  void buildBounds2D(const BoundsType &outerBound) {
    auto xExt = (outerBound[1][0] - outerBound[0][0]) / 2.;
    auto yExt = (outerBound[1][1] - outerBound[0][1]) / 2.;

    const auto BVH1 =
        csPair<csPair<T>>{outerBound[0][0], outerBound[0][1],
                          outerBound[0][0] + xExt, outerBound[0][1] + yExt};
    const auto BVH2 =
        csPair<csPair<T>>{outerBound[0][0] + xExt, outerBound[0][1],
                          uterBound[0][0] + 2 * xExt, outerBound[0][1] + yExt};
    const auto BVH3 = csPair<csPair<T>>{
        outerBound[0][0] + xExt,
        outerBound[0][1] + yExt,
        outerBound[0][0] + 2 * xExt,
        outerBound[0][1] + 2 * yExt,
    };
    const auto BVH4 = csPair<csPair<T>>{
        outerBound[0][0],
        outerBound[0][1] + yExt,
        outerBound[0][0] + xExt,
        outerBound[0][1] + 2 * yExt,
    };

    bounds = std::array<std::array<csPair<T>, 2>, 4>{BVH1, BVH2, BVH3, BVH4};
  }

  void buildBounds3D(const BoundsType &outerBound) {
    auto xExt = (outerBound[1][0] - outerBound[0][0]) / 2.;
    auto yExt = (outerBound[1][1] - outerBound[0][1]) / 2.;
    auto zExt = (outerBound[1][2] - outerBound[0][2]) / 2.;

    const auto BVH1 =
        csPair<csTriple<T>>{outerBound[0][0],        outerBound[0][1],
                            outerBound[0][2],        outerBound[0][0] + xExt,
                            outerBound[0][1] + yExt, outerBound[0][2] + zExt};
    const auto BVH2 = csPair<csTriple<T>>{
        outerBound[0][0] + xExt, outerBound[0][1],
        outerBound[0][2],        outerBound[0][0] + 2 * xExt,
        outerBound[0][1] + yExt, outerBound[0][2] + zExt};
    const auto BVH3 = csPair<csTriple<T>>{outerBound[0][0] + xExt,
                                          outerBound[0][1] + yExt,
                                          outerBound[0][2],
                                          outerBound[0][0] + 2 * xExt,
                                          outerBound[0][1] + 2 * yExt,
                                          outerBound[0][2] + zExt};
    const auto BVH4 = csPair<csTriple<T>>{outerBound[0][0],
                                          outerBound[0][1] + yExt,
                                          outerBound[0][2],
                                          outerBound[0][0] + xExt,
                                          outerBound[0][1] + 2 * yExt,
                                          outerBound[0][2] + zExt};

    // top
    const auto BVH5 = csPair<csTriple<T>>{
        outerBound[0][0],        outerBound[0][1],
        outerBound[0][2] + zExt, outerBound[0][0] + xExt,
        outerBound[0][1] + yExt, outerBound[0][2] + 2 * zExt};
    const auto BVH6 = csPair<csTriple<T>>{
        outerBound[0][0] + xExt, outerBound[0][1],
        outerBound[0][2] + zExt, outerBound[0][0] + 2 * xExt,
        outerBound[0][1] + yExt, outerBound[0][2] + 2 * zExt};
    const auto BVH7 = csPair<csTriple<T>>{
        outerBound[0][0] + xExt,     outerBound[0][1] + yExt,
        outerBound[0][2] + zExt,     outerBound[0][0] + 2 * xExt,
        outerBound[0][1] + 2 * yExt, outerBound[0][2] + 2 * zExt};
    const auto BVH8 = csPair<csTriple<T>>{outerBound[0][0],
                                          outerBound[0][1] + yExt,
                                          outerBound[0][2] + zExt,
                                          outerBound[0][0] + xExt,
                                          outerBound[0][1] + 2 * yExt,
                                          outerBound[0][2] + 2 * zExt};
    bounds = std::array<std::array<csTriple<T>, 2>, 8>{BVH1, BVH2, BVH3, BVH4,
                                                       BVH5, BVH6, BVH7, BVH8};
  }
};
