#pragma once

#include <hrleSparseIterator.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>

#include <vcKDTree.hpp>
#include <vcLogger.hpp>

namespace viennaps {

using namespace viennacore;

template <class NumericType, int D> class SurfacePointValuesToLevelSet {
  using lsDomainType = SmartPointer<viennals::Domain<NumericType, D>>;

  lsDomainType levelSet;
  SmartPointer<viennals::Mesh<NumericType>> mesh;
  std::vector<std::string> dataNames;

public:
  SurfacePointValuesToLevelSet() {}

  SurfacePointValuesToLevelSet(
      lsDomainType passedLevelSet,
      SmartPointer<viennals::Mesh<NumericType>> passedMesh,
      std::vector<std::string> passedDataNames)
      : levelSet(passedLevelSet), mesh(passedMesh), dataNames(passedDataNames) {
  }

  void setLevelSet(lsDomainType passedLevelSet) { levelSet = passedLevelSet; }

  void setMesh(SmartPointer<viennals::Mesh<NumericType>> passedMesh) {
    mesh = passedMesh;
  }

  void setDataName(std::string passesDataName) {
    dataNames.clear();
    dataNames.push_back(passesDataName);
  }

  void setDataName(std::vector<std::string> passesDataNames) {
    dataNames = passesDataNames;
  }

  void apply() {
    if (!levelSet) {
      Logger::getInstance()
          .addWarning("No level set passed to SurfacePointValuesToLevelSet.")
          .print();
      return;
    }

    if (!mesh) {
      Logger::getInstance()
          .addWarning("No mesh passed to SurfacePointValuesToLevelSet.")
          .print();
      return;
    }

    KDTree<NumericType, Vec3D<NumericType>> transTree(mesh->getNodes());
    transTree.build();
    const auto gridDelta = levelSet->getGrid().getGridDelta();

    std::vector<std::size_t> levelSetPointToMeshIds(
        levelSet->getNumberOfPoints());

    for (hrleConstSparseIterator<
             typename viennals::Domain<NumericType, D>::DomainType>
             it(levelSet->getDomain());
         !it.isFinished(); ++it) {

      if (it.isDefined()) {
        auto lsIndicies = it.getStartIndices();
        Vec3D<NumericType> levelSetPointCoordinate{0., 0., 0.};
        for (unsigned i = 0; i < D; i++) {
          levelSetPointCoordinate[i] = lsIndicies[i] * gridDelta;
        }
        auto meshPointId = transTree.findNearest(levelSetPointCoordinate);
        assert(it.getPointId() < levelSet->getNumberOfPoints());
        levelSetPointToMeshIds[it.getPointId()] = meshPointId->first;
      }
    }

    for (const auto dataName : dataNames) {
      auto pointData = mesh->getCellData().getScalarData(dataName);
      if (!pointData)
        continue;

      auto data = levelSet->getPointData().getScalarData(dataName, true);
      if (data != nullptr) {
        data->resize(levelSet->getNumberOfPoints());
      } else {
        levelSet->getPointData().insertNextScalarData(
            std::vector<NumericType>(levelSet->getNumberOfPoints()), dataName);
      }
      data = levelSet->getPointData().getScalarData(dataName);

      for (std::size_t i = 0; i < levelSetPointToMeshIds.size(); i++) {
        data->at(i) = pointData->at(levelSetPointToMeshIds[i]);
      }
    }
  }
};

} // namespace viennaps
