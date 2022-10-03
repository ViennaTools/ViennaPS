#pragma once

#include <set>

#include <csUtil.hpp>

#include <lsSmartPointer.hpp>

template <class T, int D> class csBoundingVolume {
private:
  using BVPtrType = lsSmartPointer<csBoundingVolume<T, D>>;
  using BoundsType = csPair<std::array<T, D>>;
  using CellIdsPtr = std::set<unsigned> *;

  static constexpr int numCells = 1 << D;
  std::array<std::set<unsigned>, numCells> cellIds;
  std::array<BoundsType, numCells> bounds;
  std::array<BVPtrType, numCells> links;
  int layer = -1;

public:
  csBoundingVolume() {}
  csBoundingVolume(const BoundsType &outerBound, int thisLayer)
      : layer(thisLayer) {

    if constexpr (D == 3)
      buildBounds3D(outerBound);
    else
      buildBounds2D(outerBound);

    if (layer > 0) {
      for (size_t i = 0; i < numCells; i++) {
        links[i] = BVPtrType::New(bounds[i], layer - 1);
      }
    }
  }

  CellIdsPtr getCellIds(const std::array<T, 3> &point) {
    auto vid = getVolumeIndex(point);
    // assert(vid < numCells && "Point in invalid BV");

    if (vid == numCells)
      return nullptr;

    if (layer == 0) {
      return &cellIds[vid];
    }

    return links[vid]->getCellIds(point);
  }

  void getBoundingVolumeBounds(const std::array<T, D> &point) {
    auto vid = getVolumeIndex(point);
    assert(vid < numCells && "Point in invalid BV");

    if (layer == 0) {
      printBound(vid);
      return;
    }

    links[vid]->getBoundingVolumeBounds(point);
  }

  size_t getTotalCellCounts() {
    size_t count = 0;
    if (layer == 0) {
      for (size_t i = 0; i < numCells; i++) {
        count += cellIds[i].size();
      }
      return count;
    } else {
      for (size_t i = 0; i < numCells; i++) {
        count += links[i]->getTotalCellCounts();
      }
    }
    return count;
  }

  void clear() {
    if (layer == 0) {
      for (size_t i = 0; i < numCells; i++) {
        cellIds[i].clear();
      }
    } else {
      for (size_t i = 0; i < numCells; i++) {
        links[i]->clear();
      }
    }
  }

  BVPtrType getLink(const std::array<T, 3> &point) {
    auto vid = getVolumeIndex(point);
    return getLink(vid);
  }

  BVPtrType getLink(size_t vid) { return links[vid]; }

  size_t getVolumeIndex(const std::array<T, 3> &point) {
    size_t vid = numCells;
    for (size_t idx = 0; idx < numCells; idx++) {
      if (insideVolume(point, idx)) {
        vid = idx;
        break;
      }
    }
    return vid;
  }

  bool insideVolume(const std::array<T, 3> &p, const size_t idx) const {
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
                          outerBound[0][0] + 2 * xExt, outerBound[0][1] + yExt};
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

    bounds =
        std::array<csPair<std::array<T, D>>, numCells>{BVH1, BVH2, BVH3, BVH4};
  }

  void buildBounds3D(const BoundsType &outerBound) {
    auto xExt = (outerBound[1][0] - outerBound[0][0]) / T(2);
    auto yExt = (outerBound[1][1] - outerBound[0][1]) / T(2);
    auto zExt = (outerBound[1][2] - outerBound[0][2]) / T(2);

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
    bounds = std::array<csPair<std::array<T, D>>, numCells>{
        BVH1, BVH2, BVH3, BVH4, BVH5, BVH6, BVH7, BVH8};
  }
};
