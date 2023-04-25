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
  psWriteVisualizationMesh(
      psSmartPointer<psDomain<NumericType, D>> passedDomain,
      std::string passedFileName,
      psSmartPointer<lsMaterialMap> passedMaterialMap)
      : domain(passedDomain), fileName(passedFileName),
        materialMap(passedMaterialMap) {}

  void apply() {
    lsWriteVisualizationMesh<NumericType, D> visMesh;
    visMesh.setFileName(fileName);
    int i = 0;
    for (auto ls : *domain->getLevelSets()) {
      visMesh.insertNextLevelSet(ls);
    }
    if (materialMap)
      visMesh.setMaterialMap(materialMap);
    visMesh.apply();
  }

  void setFileName(std::string passedFileName) { fileName = passedFileName; }

  void setDomain(psSmartPointer<psDomain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  void setMaterialMap(psSmartPointer<lsMaterialMap> passedMaterialMap) {
    materialMap = passedMaterialMap;
  }

private:
  psSmartPointer<psDomain<NumericType, D>> domain;
  psSmartPointer<lsMaterialMap> materialMap;
  std::string fileName;
};