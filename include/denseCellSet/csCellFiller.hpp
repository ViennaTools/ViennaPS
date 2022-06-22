#pragma once

#include "csUtil.hpp"
#include <rayRNG.hpp>

template <class T> class csCellFiller {
public:
  virtual T fill(unsigned cellId, T distance, T &energy, int materialId,
                 csTriple<T> &position, csTriple<T> &direction,
                 const T stepDistance, rayRNG &RNG) {
    return 0.;
  }

  virtual T fillArea(unsigned cellId, T distanceCovered, T normalDistance,
                     T fillStart, int materialId) {
    return 0.;
  }

  // We make clear that this class has to be inherited
protected:
  csCellFiller() = default;
  csCellFiller(const csCellFiller &) = default;
  csCellFiller(csCellFiller &&) = default;
};