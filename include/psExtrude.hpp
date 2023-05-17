#pragma once

#include <psDomain.hpp>

#include <lsExtrude.hpp>

template <class NumericType> class psExtrude {
  psSmartPointer<psDomain<NumericType, 2>> inputDomain;
  psSmartPointer<psDomain<NumericType, 3>> outputDomain;
  std::array<NumericType, 2> extent = {0., 0.};
  int extrudeDim = 0;

public:
  psExtrude() {}
  psExtrude(lsSmartPointer<psDomain<NumericType, 2>> passedInputDomain,
            lsSmartPointer<psDomain<NumericType, 3>> passedOutputDomain,
            std::array<NumericType, 2> passedExtent,
            const int passedExtrudeDim = 0)
      : inputDomain(passedInputDomain), outputDomain(passedOutputDomain),
        extent(passedExtent), extrudeDim(passedExtrudeDim) {}

  void
  setInputDomain(lsSmartPointer<psDomain<NumericType, 2>> passedInputDomain) {
    inputDomain = passedInputDomain;
  }

  // The 3D output domain will be overwritten by the extruded domain
  void setOutputDomain(
      lsSmartPointer<psDomain<NumericType, 3>> &passedOutputDomain) {
    outputDomain = passedOutputDomain;
  }

  // Set the min and max extent in the extruded dimension
  void setExtent(std::array<NumericType, 2> passedExtent) {
    extent = passedExtent;
  }

  // Set which index of the added dimension (x: 0, y: 1, z: 2)
  void setExtrudeDimension(const int passedExtrudeDim) {
    extrudeDim = passedExtrudeDim;
  }

  void apply() {
    if (inputDomain == nullptr) {
      lsMessage::getInstance()
          .addWarning("No input domain supplied to psExtrude! Not converting.")
          .print();
    }
    if (outputDomain == nullptr) {
      lsMessage::getInstance()
          .addWarning("No output domain supplied to psExtrude! Not converting.")
          .print();
      return;
    }

    auto materialMap = inputDomain->getMaterialMap();
    outputDomain->clear();

    for (std::size_t i = 0; i < inputDomain->getLevelSets()->size(); i++) {
      auto tmpLS = psSmartPointer<lsDomain<NumericType, 3>>::New();
      lsExtrude<NumericType>(inputDomain->getLevelSets()->at(i), tmpLS, extent,
                             extrudeDim)
          .apply();

      {
        auto mesh = psSmartPointer<lsMesh<NumericType>>::New();
        lsToSurfaceMesh<NumericType, 3>(tmpLS, mesh).apply();
        lsVTKWriter<NumericType>(mesh, "ext_" + std::to_string(i) + ".vtp")
            .apply();
      }

      if (materialMap) {
        auto material = materialMap->getMaterialAtIdx(i);
        outputDomain->insertNextLevelSetAsMaterial(tmpLS, material, false);
      } else {
        outputDomain->insertNextLevelSet(tmpLS, false);
      }
    }
  }
};