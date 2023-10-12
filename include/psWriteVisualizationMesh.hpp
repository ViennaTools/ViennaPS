#pragma once

#include <lsToSurfaceMesh.hpp>
#include <lsWriteVisualizationMesh.hpp>

#include <psDomain.hpp>

template <class NumericType, int D> class psWriteVisualizationMesh {
public:
  psWriteVisualizationMesh() {}
  psWriteVisualizationMesh(
      psSmartPointer<psDomain<NumericType, D>> passedDomain,
      std::string passedFileName)
      : domain(passedDomain), fileName(passedFileName) {}

  void apply() {
    lsWriteVisualizationMesh<NumericType, D> visMesh;
    visMesh.setFileName(fileName);
    int i = 0;
    for (auto ls : *domain->getLevelSets()) {
      visMesh.insertNextLevelSet(ls);
    }
    if (domain->getMaterialMap())
      visMesh.setMaterialMap(domain->getMaterialMap()->getMaterialMap());
    visMesh.apply();
  }

  void setFileName(std::string passedFileName) { fileName = passedFileName; }

  void setDomain(psSmartPointer<psDomain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

private:
  psSmartPointer<psDomain<NumericType, D>> domain;
  psSmartPointer<lsMaterialMap> materialMap;
  std::string fileName;
};