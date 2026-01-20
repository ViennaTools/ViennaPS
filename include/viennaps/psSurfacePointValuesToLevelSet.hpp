#pragma once

#include "psPreCompileMacros.hpp"

#include <hrleSparseIterator.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>

#include <utility>
#include <vcKDTree.hpp>
#include <vcLogger.hpp>

namespace viennaps {

using namespace viennacore;

VIENNAPS_TEMPLATE_ND class SurfacePointValuesToLevelSet {
  using lsDomainType = SmartPointer<viennals::Domain<NumericType, D>>;

  lsDomainType levelSet;
  SmartPointer<viennals::Mesh<NumericType>> mesh;
  std::vector<std::string> dataNames;

public:
  SurfacePointValuesToLevelSet() = default;
  SurfacePointValuesToLevelSet(
      lsDomainType passedLevelSet,
      SmartPointer<viennals::Mesh<NumericType>> passedMesh,
      std::vector<std::string> passedDataNames)
      : levelSet(passedLevelSet), mesh(passedMesh),
        dataNames(std::move(passedDataNames)) {}

  void setLevelSet(lsDomainType passedLevelSet) { levelSet = passedLevelSet; }

  void setMesh(SmartPointer<viennals::Mesh<NumericType>> passedMesh) {
    mesh = passedMesh;
  }

  void setDataName(const std::string &passesDataName) {
    dataNames.clear();
    dataNames.push_back(passesDataName);
  }

  void setDataName(std::vector<std::string> passesDataNames) {
    dataNames = std::move(passesDataNames);
  }

  void apply() {
    if (!levelSet) {
      VIENNACORE_LOG_ERROR(
          "No level set passed to SurfacePointValuesToLevelSet.");
      return;
    }

    if (!mesh) {
      VIENNACORE_LOG_ERROR("No mesh passed to SurfacePointValuesToLevelSet.");
      return;
    }

    KDTree<NumericType, Vec3D<NumericType>> transTree(mesh->getNodes());
    transTree.build();
    const auto gridDelta = levelSet->getGrid().getGridDelta();

    std::vector<std::size_t> levelSetPointToMeshIds(
        levelSet->getNumberOfPoints());

    for (viennahrle::ConstSparseIterator<
             typename viennals::Domain<NumericType, D>::DomainType>
             it(levelSet->getDomain());
         !it.isFinished(); ++it) {

      if (it.isDefined()) {
        auto lsIndices = it.getStartIndices();
        Vec3D<NumericType> levelSetPointCoordinate{0., 0., 0.};
        for (unsigned i = 0; i < D; i++) {
          levelSetPointCoordinate[i] = lsIndices[i] * gridDelta;
        }
        auto meshPointId = transTree.findNearest(levelSetPointCoordinate);
        assert(it.getPointId() < levelSet->getNumberOfPoints());
        levelSetPointToMeshIds[it.getPointId()] = meshPointId->first;
      }
    }

    for (const auto &dataName : dataNames) {
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
