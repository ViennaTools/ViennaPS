#pragma once

#include <string>
#include <vector>

namespace viennaps {

namespace gpu {

template <typename T> struct Particle {
  std::string name;
  std::vector<std::string> dataLabels;

  float sticking = 1.f;
  float cosineExponent = 1.f;

  Vec3Df direction = {0.f, 0.f, -1.f};
};

} // namespace gpu
} // namespace viennaps