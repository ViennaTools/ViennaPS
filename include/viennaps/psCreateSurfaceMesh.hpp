#pragma once

#include "psPreCompileMacros.hpp"

#include <lsDomain.hpp>
#include <lsMesh.hpp>
#include <lsToSurfaceMesh.hpp>

#include <rayMesh.hpp>

#include <vcKDTree.hpp>

namespace viennaps {

using namespace viennacore;

inline viennaray::TriangleMesh
CreateTriangleMesh(const float gridDelta,
                   const SmartPointer<viennals::Mesh<float>> &mesh) {
  assert(mesh->getCellData().getVectorData("Normals") != nullptr &&
         "Mesh normals not found in cell data under label 'Normals'.");
  viennaray::TriangleMesh triangleMesh;

  triangleMesh.gridDelta = gridDelta;
  triangleMesh.triangles = mesh->triangles;
  triangleMesh.nodes = mesh->nodes;
  triangleMesh.minimumExtent = mesh->minimumExtent;
  triangleMesh.maximumExtent = mesh->maximumExtent;
  triangleMesh.normals = *mesh->getCellData().getVectorData("Normals");

  return triangleMesh;
}

inline void CopyTriangleMesh(const float gridDelta,
                             const SmartPointer<viennals::Mesh<float>> &mesh,
                             viennaray::TriangleMesh &triangleMesh) {
  triangleMesh.gridDelta = gridDelta;
  triangleMesh.triangles = mesh->triangles;
  triangleMesh.nodes = mesh->nodes;
  triangleMesh.minimumExtent = mesh->minimumExtent;
  triangleMesh.maximumExtent = mesh->maximumExtent;
  triangleMesh.normals = *mesh->getCellData().getVectorData("Normals");
}

template <Numeric LsNT, Numeric MeshNT, int D>
  requires Dimension<D>
class CreateSurfaceMesh {

  using lsDomainType = viennals::Domain<LsNT, D>;
  using CellIteratorType = viennahrle::ConstSparseCellIterator<
      typename viennals::Domain<LsNT, D>::DomainType>;
  using kdTreeType = KDTree<LsNT, std::array<LsNT, 3>>;

  SmartPointer<lsDomainType> levelSet = nullptr;
  SmartPointer<viennals::Mesh<MeshNT>> mesh = nullptr;
  SmartPointer<kdTreeType> kdTree = nullptr;

  const MeshNT epsilon;
  MeshNT minNodeDistanceFactor = 0.05;

  struct I3 {
    int x, y, z;
    bool operator==(const I3 &o) const {
      return x == o.x && y == o.y && z == o.z;
    }
  };

  struct I3Hash {
    size_t operator()(const I3 &k) const {
      // 64-bit mix
      uint64_t a = (uint64_t)(uint32_t)k.x;
      uint64_t b = (uint64_t)(uint32_t)k.y;
      uint64_t c = (uint64_t)(uint32_t)k.z;
      uint64_t h = a * 0x9E3779B185EBCA87ULL;
      h ^= b + 0xC2B2AE3D27D4EB4FULL + (h << 6) + (h >> 2);
      h ^= c + 0x165667B19E3779F9ULL + (h << 6) + (h >> 2);
      return (size_t)h;
    }
  };

public:
  CreateSurfaceMesh(SmartPointer<lsDomainType> passedLevelSet,
                    SmartPointer<viennals::Mesh<MeshNT>> passedMesh,
                    SmartPointer<kdTreeType> passedKdTree = nullptr,
                    double eps = 1e-12, double minNodeDistFactor = 0.05)
      : levelSet(passedLevelSet), mesh(passedMesh), kdTree(passedKdTree),
        epsilon(eps), minNodeDistanceFactor(minNodeDistFactor) {}

  void apply() {
    if (levelSet == nullptr) {
      VIENNACORE_LOG_ERROR("No level set was passed to CreateSurfaceMesh.");
      return;
    }
    if (mesh == nullptr) {
      VIENNACORE_LOG_ERROR("No mesh was passed to CreateSurfaceMesh.");
      return;
    }

    mesh->clear();
    mesh->minimumExtent = Vec3D<MeshNT>{std::numeric_limits<MeshNT>::max(),
                                        std::numeric_limits<MeshNT>::max(),
                                        std::numeric_limits<MeshNT>::max()};
    mesh->maximumExtent = Vec3D<MeshNT>{std::numeric_limits<MeshNT>::lowest(),
                                        std::numeric_limits<MeshNT>::lowest(),
                                        std::numeric_limits<MeshNT>::lowest()};

    constexpr unsigned int corner0[12] = {0, 1, 2, 0, 4, 5, 6, 4, 0, 1, 3, 2};
    constexpr unsigned int corner1[12] = {1, 3, 3, 2, 5, 7, 7, 6, 4, 5, 7, 6};
    constexpr unsigned int direction[12] = {0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2};

    // test if level set function consists of at least 2 layers of
    // defined grid points
    if (levelSet->getLevelSetWidth() < 2) {
      VIENNACORE_LOG_WARNING(
          "Level-set is less than 2 layers wide. Expanding to 2 layers.");
      viennals::Expand<LsNT, D>(levelSet, 2).apply();
    }

    typedef std::map<viennahrle::Index<D>, unsigned> nodeContainerType;

    nodeContainerType nodes[D];
    const auto gridDelta = levelSet->getGrid().getGridDelta();
    const MeshNT invMinNodeDistance = 1. / (gridDelta * minNodeDistanceFactor);
    std::unordered_map<I3, unsigned, I3Hash> nodeIdByBin;

    std::vector<Vec3D<LsNT>> triangleCenters;
    std::vector<Vec3D<MeshNT>> normals;

    if constexpr (D == 3) {
      // Estimate triangle count and reserve memory
      size_t estimatedTriangles = levelSet->getDomain().getNumberOfPoints() / 4;
      triangleCenters.reserve(estimatedTriangles);
      normals.reserve(estimatedTriangles);
      mesh->triangles.reserve(estimatedTriangles);
      mesh->nodes.reserve(estimatedTriangles * 4);
      nodeIdByBin.reserve(estimatedTriangles * 4);
    }

    const bool buildKdTreeFlag = kdTree != nullptr && D == 3;
    const bool checkNodeFlag = minNodeDistanceFactor > 0;

    auto quantize = [&](const Vec3D<MeshNT> &p) -> I3 {
      return {(int)std::llround(p[0] * invMinNodeDistance),
              (int)std::llround(p[1] * invMinNodeDistance),
              (int)std::llround(p[2] * invMinNodeDistance)};
    };

    // iterate over all active surface points
    for (CellIteratorType cellIt(levelSet->getDomain()); !cellIt.isFinished();
         cellIt.next()) {

      for (int u = 0; u < D; u++) {
        while (!nodes[u].empty() &&
               nodes[u].begin()->first <
                   viennahrle::Index<D>(cellIt.getIndices()))
          nodes[u].erase(nodes[u].begin());
      }

      unsigned signs = 0;
      for (int i = 0; i < (1 << D); i++) {
        if (cellIt.getCorner(i).getValue() >= LsNT(0))
          signs |= (1 << i);
      }

      // all corners have the same sign, so no surface here
      if (signs == 0)
        continue;
      if (signs == (1 << (1 << D)) - 1)
        continue;

      // for each element
      const int *Triangles;
      if constexpr (D == 2) {
        Triangles = lsInternal::MarchingCubes::polygonize2d(signs);
      } else {
        Triangles = lsInternal::MarchingCubes::polygonize3d(signs);
      }

      for (; Triangles[0] != -1; Triangles += D) {
        std::array<unsigned, D> nod_numbers;

        // for each node
        for (int n = 0; n < D; n++) {
          const int edge = Triangles[n];

          unsigned p0 = corner0[edge];
          unsigned p1 = corner1[edge];

          // determine direction of edge
          unsigned dir = direction[edge];

          // look for existing surface node
          viennahrle::Index<D> d(cellIt.getIndices());
          auto p0B = viennahrle::BitMaskToIndex<D>(p0);
          d += p0B;

          if (auto nodeIt = nodes[dir].find(d); nodeIt != nodes[dir].end()) {
            nod_numbers[n] = nodeIt->second;
          } else {
            // if node does not exist yet
            // calculate coordinate of new node
            Vec3D<MeshNT> cc{0., 0., 0.}; // initialise with zeros
            for (int z = 0; z < D; z++) {
              if (z != dir) {
                // TODO might not need BitMaskToVector here, just check if z
                // bit is set
                cc[z] = static_cast<MeshNT>(cellIt.getIndices(z) + p0B[z]);
              } else {
                auto d0 = static_cast<MeshNT>(cellIt.getCorner(p0).getValue());
                auto d1 = static_cast<MeshNT>(cellIt.getCorner(p1).getValue());

                // calculate the surface-grid intersection point
                if (d0 == -d1) { // includes case where d0=d1=0
                  cc[z] = static_cast<MeshNT>(cellIt.getIndices(z)) + 0.5;
                } else {
                  if (std::abs(d0) <= std::abs(d1)) {
                    cc[z] = static_cast<MeshNT>(cellIt.getIndices(z)) +
                            (d0 / (d0 - d1));
                  } else {
                    cc[z] = static_cast<MeshNT>(cellIt.getIndices(z) + 1) -
                            (d1 / (d1 - d0));
                  }
                }
                cc[z] = std::max(cc[z], cellIt.getIndices(z) + epsilon);
                cc[z] = std::min(cc[z], (cellIt.getIndices(z) + 1) - epsilon);
              }
              cc[z] *= gridDelta;
            }

            int nodeIdx = -1;
            if (checkNodeFlag) {
              auto q = quantize(cc);
              auto it = nodeIdByBin.find(q);
              if (it != nodeIdByBin.end())
                nodeIdx = it->second;
            }
            if (nodeIdx >= 0) {
              nod_numbers[n] = nodeIdx;
            } else {
              // insert new node
              nod_numbers[n] = mesh->insertNextNode(cc);
              nodes[dir][d] = nod_numbers[n];
              if (checkNodeFlag)
                nodeIdByBin.emplace(quantize(cc), nod_numbers[n]);

              // update mesh extents
              for (int a = 0; a < D; a++) {
                mesh->minimumExtent[a] =
                    std::min(mesh->minimumExtent[a], cc[a]);
                mesh->maximumExtent[a] =
                    std::max(mesh->maximumExtent[a], cc[a]);
              }
            }
          }
        }

        if (!triangleMisformed(nod_numbers)) {
          auto normal = calculateNormal(mesh->nodes, nod_numbers);
          auto n2 = Norm2(normal);
          if (n2 > epsilon) {
            MeshNT invn = static_cast<MeshNT>(1. / std::sqrt(n2));
            for (int d = 0; d < D; d++) {
              normal[d] *= invn;
            }
            normals.push_back(normal);
            mesh->insertNextElement(nod_numbers); // insert new surface element

            if (buildKdTreeFlag) {
              triangleCenters.push_back(
                  {static_cast<LsNT>((mesh->nodes[nod_numbers[0]][0] +
                                      mesh->nodes[nod_numbers[1]][0] +
                                      mesh->nodes[nod_numbers[2]][0]) /
                                     3.),
                   static_cast<LsNT>((mesh->nodes[nod_numbers[0]][1] +
                                      mesh->nodes[nod_numbers[1]][1] +
                                      mesh->nodes[nod_numbers[2]][1]) /
                                     3.),
                   static_cast<LsNT>((mesh->nodes[nod_numbers[0]][2] +
                                      mesh->nodes[nod_numbers[1]][2] +
                                      mesh->nodes[nod_numbers[2]][2]) /
                                     3.)});
            }
          }
        }
      }
    }

    mesh->cellData.insertNextVectorData(normals, "Normals");
    mesh->nodes.shrink_to_fit();
    mesh->triangles.shrink_to_fit();

    if (buildKdTreeFlag) {
      kdTree->setPoints(triangleCenters);
      kdTree->build();
    }
  }

private:
  static inline bool
  triangleMisformed(const std::array<unsigned, D> &nod_numbers) noexcept {
    if constexpr (D == 2) {
      return nod_numbers[0] == nod_numbers[1];
    } else {
      return nod_numbers[0] == nod_numbers[1] ||
             nod_numbers[0] == nod_numbers[2] ||
             nod_numbers[1] == nod_numbers[2];
    }
  }

  static inline Vec3D<MeshNT>
  calculateNormal(const std::vector<Vec3D<MeshNT>> &nodes,
                  const std::array<unsigned, D> &nod_numbers) noexcept {
    if constexpr (D == 2) {
      auto const &p0 = nodes[nod_numbers[0]];
      auto const &p1 = nodes[nod_numbers[1]];
      return Vec3D<MeshNT>{-(p1[1] - p0[1]), (p1[0] - p0[0]), MeshNT(0)};
    } else {
      return CrossProduct(nodes[nod_numbers[1]] - nodes[nod_numbers[0]],
                          nodes[nod_numbers[2]] - nodes[nod_numbers[0]]);
    }
  }
};

} // namespace viennaps