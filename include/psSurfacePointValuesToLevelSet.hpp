#pragma once

#include <lsDomain.hpp>

#include <hrleSparseIterator.hpp>

#include <psKDTree.hpp>
#include <psSmartPointer.hpp>

template <class NumericType, int D> class psSurfacePointValuesToLevelSet {
  using lsDomainType = psSmartPointer<lsDomain<NumericType, D>>;

  lsDomainType levelSet;
  psSmartPointer<lsMesh<NumericType>> mesh;
  std::vector<std::string> dataNames;

public:
  psSurfacePointValuesToLevelSet() {}

  psSurfacePointValuesToLevelSet(lsDomainType passedLevelSet,
                                 psSmartPointer<lsMesh<NumericType>> passedMesh,
                                 std::vector<std::string> passedDataNames)
      : levelSet(passedLevelSet), mesh(passedMesh), dataNames(passedDataNames) {
  }

  void setLevelSet(lsDomainType passedLevelSet) { levelSet = passedLevelSet; }

  void setMesh(psSmartPointer<lsMesh<NumericType>> passedMesh) {
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
      psLogger::getInstance()
          .addWarning("No level set passed to psSurfacePointValuesToLevelSet.")
          .print();
      return;
    }

    if (!mesh) {
      psLogger::getInstance()
          .addWarning("No mesh passed to psSurfacePointValuesToLevelSet.")
          .print();
      return;
    }

    psKDTree<NumericType, std::array<NumericType, 3>> transTree(
        mesh->getNodes());
    transTree.build();
    const auto gridDelta = levelSet->getGrid().getGridDelta();

    std::vector<std::size_t> levelSetPointToMeshIds(

        levelSet->getNumberOfPoints());

    for (hrleConstSparseIterator<typename lsDomain<NumericType, D>::DomainType>
             it(levelSet->getDomain());
         !it.isFinished(); ++it) {

      if (it.isDefined()) {
        auto lsIndicies = it.getStartIndices();
        std::array<NumericType, 3> levelSetPointCoordinate{0., 0., 0.};
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
      if (!pointData) {
        psLogger::getInstance()
            .addWarning("Could not find " + dataName + " in mesh values.")
            .print();
        continue;
      }
      auto data = levelSet->getPointData().getScalarData(dataName);
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