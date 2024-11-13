#pragma once

#include <vcVectorUtil.hpp>

#include <vector>

namespace viennaps {

namespace gpu {

using namespace viennacore;

struct TriangleMesh {
  std::vector<Vec3Df> vertex;
  std::vector<Vec3D<int>> index;

  Vec3Df minCoords;
  Vec3Df maxCoords;
  float gridDelta;
};

struct SphereMesh {
  std::vector<Vec3Df> vertex;
  std::vector<float> radius;

  Vec3Df minCoords;
  Vec3Df maxCoords;
  float gridDelta;
};

struct OrientedPointCloud {
  std::vector<Vec3Df> vertex;
  std::vector<Vec3Df> normal;
};

} // namespace gpu
} // namespace viennaps