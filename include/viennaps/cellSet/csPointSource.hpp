#pragma once

#include <raySource.hpp>

namespace viennacs {

using namespace viennacore;

template <typename NumericType, int D>
class PointSource : public viennaray::Source<NumericType> {
  const unsigned mNumPoints;
  const csTriple<NumericType> origin;
  const csTriple<NumericType> direction;

public:
  PointSource(csTriple<NumericType> passedOrigin,
              csTriple<NumericType> passedDirection,
              std::array<int, 5> &pTraceSettings, const size_t pNumPoints)
      : origin(passedOrigin), direction(passedDirection),
        mNumPoints(pNumPoints) {}

  Pair<csTriple<NumericType>>
  getOriginAndDirection(const size_t idx, RNG &rngState) const override {
    return {origin, direction};
  }

  size_t getNumPoints() const override { return mNumPoints; }
};

} // namespace viennacs
