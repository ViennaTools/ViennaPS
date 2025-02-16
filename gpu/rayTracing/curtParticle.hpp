#pragma once

#include <string>
#include <vector>

namespace viennaps {

namespace gpu {

template <typename T> struct Particle {
  std::string name;
  std::vector<std::string> dataLabels;

  T sticking = 1.;
  T cosineExponent = 1.;

  Vec3D<T> direction = {0., 0., -1.0};
};

} // namespace gpu
} // namespace viennaps