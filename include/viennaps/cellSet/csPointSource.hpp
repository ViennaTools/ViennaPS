#pragma once

#include <raySource.hpp>

namespace viennacs {

using namespace viennacore;

template <typename NumericType, int D>
class PointSource : public viennaray::Source<NumericType> {
  const unsigned mNumPoints;
  const Triple<NumericType> origin;
  const Triple<NumericType> direction;

public:
  PointSource(Triple<NumericType> passedOrigin,
              Triple<NumericType> passedDirection,
              std::array<int, 5> &pTraceSettings, const size_t pNumPoints)
      : origin(passedOrigin), direction(passedDirection),
        mNumPoints(pNumPoints) {}

  Pair<Triple<NumericType>>
  getOriginAndDirection(const size_t idx,
                        viennaray::RNG &RngState) const override {
    return {origin, direction};
  }

  size_t getNumPoints() const override { return mNumPoints; }
};

} // namespace viennacs
