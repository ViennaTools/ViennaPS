#pragma once

#include "csUtil.hpp"

#include <raySource.hpp>

template <typename NumericType, int D>
class csPointSource : public raySource<NumericType> {
  const unsigned mNumPoints;
  const csTriple<NumericType> origin;
  const csTriple<NumericType> direction;

public:
  csPointSource(csTriple<NumericType> passedOrigin,
                csTriple<NumericType> passedDirection,
                std::array<int, 5> &pTraceSettings, const size_t pNumPoints)
      : origin(passedOrigin), direction(passedDirection),
        mNumPoints(pNumPoints) {}

  rayPair<rayTriple<NumericType>>
  getOriginAndDirection(const size_t idx, rayRNG &RngState) const override {
    return {origin, direction};
  }

  size_t getNumPoints() const override { return mNumPoints; }
};
