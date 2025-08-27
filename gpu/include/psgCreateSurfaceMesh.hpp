#pragma once

#include <lsDomain.hpp>
#include <lsMesh.hpp>
#include <lsToSurfaceMesh.hpp>

#include <raygMesh.hpp>

#include <vcKDTree.hpp>

namespace viennaps::gpu {

using namespace viennacore;

viennaray::gpu::TriangleMesh
CreateTriangleMesh(const float gridDelta,
                   SmartPointer<viennals::Mesh<float>> &mesh) {
  viennaray::gpu::TriangleMesh triangleMesh;

  // if constexpr (std::is_same_v<NumericType, float>) {
  triangleMesh.gridDelta = gridDelta;
  triangleMesh.vertices = mesh->nodes;
  triangleMesh.minimumExtent = mesh->minimumExtent;
  triangleMesh.maximumExtent = mesh->maximumExtent;
  //   } else {
  //     triangleMesh.gridDelta = static_cast<float>(gridDelta);
  //     const auto &nodes = mesh->nodes;
  //     triangleMesh.vertices.resize(nodes.size());
  // #pragma omp parallel for schedule(static)
  //     for (ptrdiff_t i = 0; i < (ptrdiff_t)nodes.size(); i++) {
  // #pragma omp simd
  //       for (int d = 0; d < 3; d++)
  //         triangleMesh.vertices[i][d] = static_cast<float>(nodes[i][d]);
  //     }
  //     triangleMesh.minimumExtent =
  //     {static_cast<float>(mesh->minimumExtent[0]),
  //                                   static_cast<float>(mesh->minimumExtent[1]),
  //                                   static_cast<float>(mesh->minimumExtent[2])};
  //     triangleMesh.maximumExtent =
  //     {static_cast<float>(mesh->maximumExtent[0]),
  //                                   static_cast<float>(mesh->maximumExtent[1]),
  //                                   static_cast<float>(mesh->maximumExtent[2])};
  //   }

  triangleMesh.triangles = mesh->triangles;

  return triangleMesh;
}

template <class LsNT, class MeshNT = LsNT, int D = 3> class CreateSurfaceMesh {

  typedef viennals::Domain<LsNT, D> lsDomainType;
  typedef typename viennals::Domain<LsNT, D>::DomainType hrleDomainType;
  typedef KDTree<LsNT, std::array<LsNT, 3>> kdTreeType;

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
      Logger::getInstance()
          .addError("No level set was passed to CreateSurfaceMesh.")
          .print();
      return;
    }
    if (mesh == nullptr) {
      Logger::getInstance()
          .addError("No mesh was passed to CreateSurfaceMesh.")
          .print();
      return;
    }

    mesh->clear();
    const auto gridDelta = levelSet->getGrid().getGridDelta();
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
      Logger::getInstance()
          .addWarning(
              "Level-set is less than 2 layers wide. Expanding to 2 layers.")
          .print();
      viennals::Expand<LsNT, D>(levelSet, 2).apply();
    }

    typedef std::map<viennahrle::Index<D>, unsigned> nodeContainerType;

    nodeContainerType nodes[D];
    const MeshNT minNodeDistance = gridDelta * minNodeDistanceFactor;
    std::unordered_map<I3, unsigned, I3Hash> nodeIdByBin;

    typename nodeContainerType::iterator nodeIt;

    std::vector<Vec3D<LsNT>> triangleCenters;
    std::vector<Vec3D<MeshNT>> normals;

    // Estimate triangle count and reserve memory
    size_t estimatedTriangles = levelSet->getDomain().getNumberOfPoints() / 4;
    triangleCenters.reserve(estimatedTriangles);
    normals.reserve(estimatedTriangles);
    mesh->triangles.reserve(estimatedTriangles);
    mesh->nodes.reserve(estimatedTriangles * 4);
    nodeIdByBin.reserve(estimatedTriangles * 4);

    const bool buildKdTreeFlag = kdTree != nullptr;
    const bool checkNodeFlag = minNodeDistanceFactor > 0;

    auto quantize = [&](const Vec3D<MeshNT> &p) -> I3 {
      const MeshNT inv = MeshNT(1) / minNodeDistance;
      return {(int)std::llround(p[0] * inv), (int)std::llround(p[1] * inv),
              (int)std::llround(p[2] * inv)};
    };

    // iterate over all active surface points
    for (viennahrle::ConstSparseCellIterator<hrleDomainType> cellIt(
             levelSet->getDomain());
         !cellIt.isFinished(); cellIt.next()) {

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

          nodeIt = nodes[dir].find(d);
          if (nodeIt != nodes[dir].end()) {
            nod_numbers[n] = nodeIt->second;
          } else {
            // if node does not exist yet
            // calculate coordinate of new node
            Vec3D<MeshNT> cc{}; // initialise with zeros
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
              cc[z] = gridDelta * cc[z];
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
              nod_numbers[n] =
                  mesh->insertNextNode(cc); // insert new surface node
              nodes[dir][d] = nod_numbers[n];
              if (checkNodeFlag)
                nodeIdByBin.emplace(quantize(cc), nod_numbers[n]);

              for (int a = 0; a < D; a++) {
                if (cc[a] < mesh->minimumExtent[a])
                  mesh->minimumExtent[a] = cc[a];
                if (cc[a] > mesh->maximumExtent[a])
                  mesh->maximumExtent[a] = cc[a];
              }
            }
          }
        }

        if (!triangleMisformed(nod_numbers)) {
          auto normal = calculateNormal(mesh->nodes[nod_numbers[0]],
                                        mesh->nodes[nod_numbers[1]],
                                        mesh->nodes[nod_numbers[2]]);
          auto n2 = normal[0] * normal[0] + normal[1] * normal[1] +
                    normal[2] * normal[2];
          if (n2 > epsilon) {
            mesh->insertNextElement(nod_numbers); // insert new surface element
            MeshNT invn =
                static_cast<MeshNT>(1.) / std::sqrt(static_cast<MeshNT>(n2));
            for (int d = 0; d < D; d++) {
              normal[d] *= invn;
            }
            normals.push_back(normal);

            if (buildKdTreeFlag) {
              triangleCenters.push_back(
                  {static_cast<LsNT>(mesh->nodes[nod_numbers[0]][0] +
                                     mesh->nodes[nod_numbers[1]][0] +
                                     mesh->nodes[nod_numbers[2]][0]) /
                       static_cast<LsNT>(3.),
                   static_cast<LsNT>(mesh->nodes[nod_numbers[0]][1] +
                                     mesh->nodes[nod_numbers[1]][1] +
                                     mesh->nodes[nod_numbers[2]][1]) /
                       static_cast<LsNT>(3.),
                   static_cast<LsNT>(mesh->nodes[nod_numbers[0]][2] +
                                     mesh->nodes[nod_numbers[1]][2] +
                                     mesh->nodes[nod_numbers[2]][2]) /
                       static_cast<LsNT>(3.)});
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
    if constexpr (D == 3) {
      return nod_numbers[0] == nod_numbers[1] ||
             nod_numbers[0] == nod_numbers[2] ||
             nod_numbers[1] == nod_numbers[2];
    } else {
      return nod_numbers[0] == nod_numbers[1];
    }
  }

  static inline Vec3D<MeshNT>
  calculateNormal(const Vec3D<MeshNT> &nodeA, const Vec3D<MeshNT> &nodeB,
                  const Vec3D<MeshNT> &nodeC) noexcept {
    return CrossProduct(nodeB - nodeA, nodeC - nodeA);
  }
};

} // namespace viennaps::gpu