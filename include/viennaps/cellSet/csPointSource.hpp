#pragma once

#include "csUtil.hpp"

#include <raySource.hpp>

template <typename NumericType, int D>
class csPointSource : public raySource<csPointSource<NumericType, D>> {
  const unsigned mNumPoints;
  const csTriple<NumericType> origin;
  const csTriple<NumericType> direction;

public:
  csPointSource(csTriple<NumericType> passedOrigin,
                csTriple<NumericType> passedDirection,
                std::array<int, 5> &pTraceSettings, const size_t pNumPoints)
      : origin(passedOrigin), direction(passedDirection),
        mNumPoints(pNumPoints) {}

  void fillRay(RTCRay &ray, const size_t idx, rayRNG &RngState) const {
#ifdef ARCH_X86
    reinterpret_cast<__m128 &>(ray) =
        _mm_set_ps(1e-4f, (float)origin[2], (float)origin[1], (float)origin[0]);

    reinterpret_cast<__m128 &>(ray.dir_x) = _mm_set_ps(
        0.0f, (float)direction[2], (float)direction[1], (float)direction[0]);
#else
    ray.org_x = (float)origin[0];
    ray.org_y = (float)origin[1];
    ray.org_z = (float)origin[2];
    ray.tnear = 1e-4f;

    ray.dir_x = (float)direction[0];
    ray.dir_y = (float)direction[1];
    ray.dir_z = (float)direction[2];
    ray.time = 0.0f;
#endif
  }

  size_t getNumPoints() const { return mNumPoints; }
};
