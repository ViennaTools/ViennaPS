#pragma once

#include <vcSmartPointer.hpp>
#include <vcVectorUtil.hpp>

#include <lsMesh.hpp>

#include <vector>

namespace viennaps {

namespace gpu {

using namespace viennacore;

struct LineMesh {
  std::vector<Vec3Df> vertices;
  std::vector<Vec2D<unsigned>> lines;

  Vec3Df minimumExtent;
  Vec3Df maximumExtent;
  float gridDelta;
};

template <typename NumericType> struct TriangleMesh {
  TriangleMesh() = default;
  TriangleMesh(float gd, SmartPointer<viennals::Mesh<NumericType>> mesh)
      : gridDelta(gd), triangles(mesh->triangles) {
    if constexpr (std::is_same_v<NumericType, float>) {
      vertices = mesh->nodes;
      minimumExtent = mesh->minimumExtent;
      maximumExtent = mesh->maximumExtent;
    } else {
      vertices.reserve(mesh->nodes.size());
      for (const auto &node : mesh->nodes) {
        vertices.push_back({static_cast<float>(node[0]),
                            static_cast<float>(node[1]),
                            static_cast<float>(node[2])});
      }
      minimumExtent = {static_cast<float>(mesh->minimumExtent[0]),
                       static_cast<float>(mesh->minimumExtent[1]),
                       static_cast<float>(mesh->minimumExtent[2])};
      maximumExtent = {static_cast<float>(mesh->maximumExtent[0]),
                       static_cast<float>(mesh->maximumExtent[1]),
                       static_cast<float>(mesh->maximumExtent[2])};
    }
  }

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