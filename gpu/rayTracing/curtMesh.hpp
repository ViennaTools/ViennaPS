#pragma once

#include <vcVectorUtil.hpp>

#include <vector>

namespace viennaps {

namespace gpu {

using namespace viennacore;

struct TriangleMesh {
  std::vector<Vec3Df> vertices;
  std::vector<Vec3D<unsigned>> triangles;

  Vec3Df minimumExtent;
  Vec3Df maximumExtent;
  float gridDelta;
};

struct SphereMesh {
  std::vector<Vec3Df> vertices;
  std::vector<float> radii;

  Vec3Df minimumExtent;
  Vec3Df maximumExtent;
  float gridDelta;
};

struct OrientedPointCloud {
  std::vector<Vec3Df> vertices;
  std::vector<Vec3Df> normals;

  Vec3Df minimumExtent;
  Vec3Df maximumExtent;
  float gridDelta;
};

} // namespace gpu
} // namespace viennaps