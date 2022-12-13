#pragma once

#include <utGDT.hpp>

struct TriangleMesh {
  std::vector<gdt::vec3f> vertex;
  std::vector<gdt::vec3i> index;

  gdt::vec3f minCoords;
  gdt::vec3f maxCoords;
  float gridDelta;
};

struct SphereMesh {
  std::vector<gdt::vec3f> vertex;
  std::vector<float> radius;

  gdt::vec3f minCoords;
  gdt::vec3f maxCoords;
  float gridDelta;
};

struct OrientedPointCloud {
  std::vector<gdt::vec3f> vertex;
  std::vector<gdt::vec3f> normal;
};