#ifndef CS_SIMPLE_CONVERSION
#define CS_SIMPLE_CONVERSION

#include <hrleSparseStarIterator.hpp>

template <class T, int D> class csSimpleConversion {
  hrleConstSparseStarIterator<hrleDomain<T, D>> &neighborIterator;

public:
  csSimpleConversion(hrleConstSparseStarIterator<hrleDomain<T, D>> &iterator)
      : neighborIterator(iterator) {}

  float getFillingFraction() {
    return 0.5 - neighborIterator.getCenter().getValue();
  }
};

#endif