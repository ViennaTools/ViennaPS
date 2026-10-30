#pragma once

#include "psProcessContext.hpp"

#include <lsMesh.hpp>

#include <limits>
#include <vector>

namespace viennaps {

using namespace viennacore;

// The dual contouring mesh of a Euclidean surface, which the triangle flux
// engines trace, with the material behind each element; elements without
// area, which dual contouring makes where three vertices lie on one sharp
// edge, are left out, since the flux of an element is its number of hits per
// area. In 3D the element centres go into the kd-tree, in 2D the engines
// build it from the triangles the lines become.
template <class NumericType, int D, class KDTree>
void dualContouringSurface(ProcessContext<NumericType, D> &context,
                           SmartPointer<viennals::Mesh<float>> &surfaceMesh,
                           SmartPointer<KDTree> &elementKdTree,
                           std::vector<int> &materialIds) {
  const auto mesh = context.domain->getSurfaceMesh(false);
  const auto &nodes = mesh->getNodes();
  const auto *normals = mesh->getNormals();
  const auto *materials = mesh->getMaterialIds();
  const NumericType gridDelta = context.domain->getGridDelta();
  assert(normals != nullptr && materials != nullptr);

  surfaceMesh->clear();
  surfaceMesh->minimumExtent = Vec3D<float>{std::numeric_limits<float>::max(),
                                            std::numeric_limits<float>::max(),
                                            std::numeric_limits<float>::max()};
  surfaceMesh->maximumExtent =
      Vec3D<float>{std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest()};
  materialIds.clear();
  std::vector<Vec3D<float>> elementNormals;
  std::vector<Vec3D<NumericType>> centers;
  std::vector<long> nodeIds(nodes.size(), -1);
  const auto &elements = mesh->template getElements<D>();
  for (std::size_t e = 0; e < elements.size(); ++e) {
    const auto &element = elements[e];
    if constexpr (D == 2) {
      if (Norm(nodes[element[1]] - nodes[element[0]]) <=
          NumericType(1e-10) * gridDelta)
        continue;
    } else {
      NumericType longest = 0.;
      for (int k = 0; k < 3; ++k)
        longest = std::max(
            longest, Norm(nodes[element[(k + 1) % 3]] - nodes[element[k]]));
      const NumericType twiceArea =
          Norm(CrossProduct(nodes[element[1]] - nodes[element[0]],
                            nodes[element[2]] - nodes[element[0]]));
      if (twiceArea <= NumericType(1e-10) * longest * longest)
        continue;
    }
    std::array<unsigned, D> ids{};
    Vec3D<NumericType> center{0., 0., 0.};
    for (int k = 0; k < D; ++k) {
      const auto node = element[k];
      if (nodeIds[node] < 0) {
        const Vec3D<float> point{static_cast<float>(nodes[node][0]),
                                 static_cast<float>(nodes[node][1]),
                                 static_cast<float>(nodes[node][2])};
        nodeIds[node] = surfaceMesh->insertNextNode(point);
        for (int a = 0; a < D; ++a) {
          surfaceMesh->minimumExtent[a] =
              std::min(surfaceMesh->minimumExtent[a], point[a]);
          surfaceMesh->maximumExtent[a] =
              std::max(surfaceMesh->maximumExtent[a], point[a]);
        }
      }
      ids[k] = static_cast<unsigned>(nodeIds[node]);
      center = center + nodes[node] / static_cast<NumericType>(D);
    }
    surfaceMesh->insertNextElement(ids);
    const auto &normal = (*normals)[e];
    elementNormals.push_back(Vec3D<float>{static_cast<float>(normal[0]),
                                          static_cast<float>(normal[1]),
                                          static_cast<float>(normal[2])});
    materialIds.push_back(static_cast<int>((*materials)[e]));
    centers.push_back(center);
  }
  surfaceMesh->cellData.insertNextVectorData(elementNormals, "Normals");
  if constexpr (D == 3) {
    elementKdTree->setPoints(centers);
    elementKdTree->build();
  }
}

} // namespace viennaps
