#pragma once

#include <lsDomain.hpp>
#include <lsMesh.hpp>
#include <lsToSurfaceMesh.hpp>

#include <gpu/raygMesh.hpp>

namespace viennaps::gpu {

using namespace viennacore;

template <class NumericType>
viennaray::gpu::TriangleMesh<float>
CreateTriangleMesh(const NumericType gridDelta,
                   SmartPointer<viennals::Mesh<float>> mesh) {
  viennaray::gpu::TriangleMesh<float> triangleMesh;
  triangleMesh.gridDelta = static_cast<float>(gridDelta);

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

  bool checkNodeDistance = true;

  struct compareNodes {
    bool operator()(const Vec3D<MeshNT> &nodeA,
                    const Vec3D<MeshNT> &nodeB) const {
      if (nodeA[0] < nodeB[0] - minNodeDistance)
        return true;
      if (nodeA[0] > nodeB[0] + minNodeDistance)
        return false;

      if (nodeA[1] < nodeB[1] - minNodeDistance)
        return true;
      if (nodeA[1] > nodeB[1] + minNodeDistance)
        return false;

      if (nodeA[2] < nodeB[2] - minNodeDistance)
        return true;
      if (nodeA[2] > nodeB[2] + minNodeDistance)
        return false;

      return false;
    }
  };

public:
  static MeshNT minNodeDistance;

  CreateSurfaceMesh(SmartPointer<lsDomainType> passedLevelSet,
                    SmartPointer<viennals::Mesh<MeshNT>> passedMesh,
                    SmartPointer<kdTreeType> passedKdTree = nullptr,
                    double eps = 1e-12, bool checkNodeDist = true)
      : levelSet(passedLevelSet), mesh(passedMesh), kdTree(passedKdTree),
        epsilon(eps), checkNodeDistance(checkNodeDist) {}

  void apply() {
    if (levelSet == nullptr) {
      Logger::getInstance()
          .addWarning("No level set was passed to CreateSurfaceMesh.")
          .print();
      return;
    }
    if (mesh == nullptr) {
      Logger::getInstance()
          .addWarning("No mesh was passed to CreateSurfaceMesh.")
          .print();
      return;
    }

    mesh->clear();
    const auto gridDelta = levelSet->getGrid().getGridDelta();
    mesh->minimumExtent = Vec3D<MeshNT>{std::numeric_limits<MeshNT>::max(),
                                        std::numeric_limits<MeshNT>::max(),
                                        std::numeric_limits<MeshNT>::max()};
    mesh->maximumExtent = Vec3D<MeshNT>{std::numeric_limits<MeshNT>::min(),
                                        std::numeric_limits<MeshNT>::min(),
                                        std::numeric_limits<MeshNT>::min()};

    const unsigned int corner0[12] = {0, 1, 2, 0, 4, 5, 6, 4, 0, 1, 3, 2};
    const unsigned int corner1[12] = {1, 3, 3, 2, 5, 7, 7, 6, 4, 5, 7, 6};
    const unsigned int direction[12] = {0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2};

    // test if level set function consists of at least 2 layers of
    // defined grid points
    if (levelSet->getLevelSetWidth() < 2) {
      Logger::getInstance()
          .addWarning("Levelset is less than 2 layers wide. Export might fail!")
          .print();
    }

    typedef std::map<hrleVectorType<hrleIndexType, D>, unsigned>
        nodeContainerType;

    nodeContainerType nodes[D];
    minNodeDistance = gridDelta * 0.01;
    std::map<Vec3D<MeshNT>, unsigned, compareNodes> nodeCoordinates;

    typename nodeContainerType::iterator nodeIt;

    std::vector<Vec3D<LsNT>> triangleCenters;
    std::vector<Vec3D<MeshNT>> normals;
    const bool buildKdTreeFlag = kdTree != nullptr;
    const bool checkNodeFlag = checkNodeDistance;

    // iterate over all active surface points
    for (hrleConstSparseCellIterator<hrleDomainType> cellIt(
             levelSet->getDomain());
         !cellIt.isFinished(); cellIt.next()) {

      for (int u = 0; u < D; u++) {
        while (!nodes[u].empty() &&
               nodes[u].begin()->first <
                   hrleVectorType<hrleIndexType, D>(cellIt.getIndices()))
          nodes[u].erase(nodes[u].begin());
      }

      unsigned signs = 0;
      for (int i = 0; i < (1 << D); i++) {
        if (cellIt.getCorner(i).getValue() >= MeshNT(0))
          signs |= (1 << i);
      }

      // all corners have the same sign, so no surface here
      if (signs == 0)
        continue;
      if (signs == (1 << (1 << D)) - 1)
        continue;

      // for each element
      const int *Triangles =
          (D == 2) ? lsInternal::MarchingCubes::polygonize2d(signs)
                   : lsInternal::MarchingCubes::polygonize3d(signs);

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
          hrleVectorType<hrleIndexType, D> d(cellIt.getIndices());
          d += BitMaskToVector<D, hrleIndexType>(p0);

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
                cc[z] = static_cast<MeshNT>(
                    cellIt.getIndices(z) +
                    BitMaskToVector<D, hrleIndexType>(p0)[z]);
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
              auto checkNode = nodeCoordinates.find(cc);
              if (checkNode != nodeCoordinates.end()) {
                nodeIdx = checkNode->second;
              }
            }
            if (nodeIdx >= 0) {
              nod_numbers[n] = nodeIdx;
            } else {
              // insert new node
              nod_numbers[n] =
                  mesh->insertNextNode(cc); // insert new surface node
              nodes[dir][d] = nod_numbers[n];
              nodeCoordinates[cc] = nod_numbers[n];

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
          LsNT norm = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                                normal[2] * normal[2]);
          if (norm > epsilon) {
            mesh->insertNextElement(nod_numbers); // insert new surface element
            for (int d = 0; d < D; d++) {
              normal[d] /= norm;
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

    if (buildKdTreeFlag) {
      kdTree->setPoints(triangleCenters);
      kdTree->build();
    }
  }

  static bool inline triangleMisformed(
      const std::array<unsigned, D> &nod_numbers) {
    if constexpr (D == 3) {
      return nod_numbers[0] == nod_numbers[1] ||
             nod_numbers[0] == nod_numbers[2] ||
             nod_numbers[1] == nod_numbers[2];
    } else {
      return nod_numbers[0] == nod_numbers[1];
    }
  }

  Vec3D<MeshNT> calculateNormal(const Vec3D<MeshNT> &nodeA,
                                const Vec3D<MeshNT> &nodeB,
                                const Vec3D<MeshNT> &nodeC) {
    Vec3D<MeshNT> U{nodeB[0] - nodeA[0], nodeB[1] - nodeA[1],
                    nodeB[2] - nodeA[2]};
    Vec3D<MeshNT> V{nodeC[0] - nodeA[0], nodeC[1] - nodeA[1],
                    nodeC[2] - nodeA[2]};
    return {U[1] * V[2] - U[2] * V[1], U[2] * V[0] - U[0] * V[2],
            U[0] * V[1] - U[1] * V[0]};
  }
};

template <class LsNT, class MeshNT, int D>
MeshNT CreateSurfaceMesh<LsNT, MeshNT, D>::minNodeDistance = 0.1;

} // namespace viennaps::gpu