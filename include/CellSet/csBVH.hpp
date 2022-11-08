#pragma once

#include <csBoundingVolume.hpp>

#include <lsSmartPointer.hpp>

template <class T, int D> class csBVH {
private:
  using BVPtrType = lsSmartPointer<csBoundingVolume<T, D>>;
  using BoundsType = csPair<std::array<T, D>>;
  using CellIdsPtr = std::set<unsigned> *;

  unsigned numLayers = 1;
  BVPtrType BV = nullptr;

public:
  csBVH(const BoundsType &domainBounds, unsigned layers = 1)
      : numLayers(layers) {
    BV = BVPtrType::New(domainBounds, numLayers - 1);
  }

  BVPtrType getTopBV() { return BV; }

  void getLowestBVBounds(const std::array<T, 3> &point) {
    BV->getBoundingVolumeBounds(point);
  }

  CellIdsPtr getCellIds(const std::array<T, 3> &point) {
    return BV->getCellIds(point);
  }

  void clearCellIds() { BV->clear(); }

  size_t getTotalCellCount() { return BV->getTotalCellCounts(); }
};