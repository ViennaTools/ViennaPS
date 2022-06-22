#pragma once

#include "csBoundingVolume.hpp"
#include <lsSmartPointer.hpp>

template <class T> class csBVH {
private:
  using BVPtrType = lsSmartPointer<csBoundingVolume<T>>;
  using BoundsType = csPair<csTriple<T>>;
  using CellIdsPtr = std::set<unsigned> *;

  unsigned numLayers = 1;
  BVPtrType BV = nullptr;

public:
  csBVH(const BoundsType &domainBounds, unsigned layers = 1)
      : numLayers(layers) {
    BV = BVPtrType::New(domainBounds, numLayers - 1);
  }

  BVPtrType getTopBV() { return BV; }

  void getLowestBV(const csTriple<T> &point) {
    BV->getBoundingVolumeBounds(point);
  }

  CellIdsPtr getCellIds(const csTriple<T> &point) {
    return BV->getCellIds(point);
  }

  CellIdsPtr getCellIds(const T x, const T y, const T z) {
    return BV->getCellIds(csTriple<T>{x, y, z});
  }

  void clearCellIds() { BV->clear(); }

  size_t getTotalCellCount() { return BV->getTotalCellCounts(); }
};