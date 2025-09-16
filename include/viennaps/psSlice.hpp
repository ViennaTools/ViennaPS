#pragma once

#include "psDomain.hpp"
#include "psPreCompileMacros.hpp"

#include <lsSlice.hpp>
#include <lsToMesh.hpp>

namespace viennaps {
using namespace viennacore;

template <class NumericType> class Slice {
  SmartPointer<Domain<NumericType, 3>> inputDomain;
  SmartPointer<Domain<NumericType, 2>> outputDomain;
  int sliceDimension = 0;
  NumericType slicePosition = 0;
  bool reflectX = false;

public:
  Slice() = default;
  Slice(SmartPointer<Domain<NumericType, 3>> &passedInputDomain,
        SmartPointer<Domain<NumericType, 2>> &passedOutputDomain,
        int passedSliceDimension = 0, NumericType passedSlicePosition = 0)
      : inputDomain(passedInputDomain), outputDomain(passedOutputDomain),
        sliceDimension(passedSliceDimension),
        slicePosition(passedSlicePosition) {}

  void setInputDomain(SmartPointer<Domain<NumericType, 3>> passedInputDomain) {
    inputDomain = passedInputDomain;
  }

  // The 3D output domain will be overwritten by the Sliced domain
  void
  setOutputDomain(SmartPointer<Domain<NumericType, 2>> &passedOutputDomain) {
    outputDomain = passedOutputDomain;
  }

  // Set the min and max extent in the Sliced dimension
  void setSliceDimension(int passedSliceDimension) {
    sliceDimension = passedSliceDimension;
  }

  void setSlicePosition(NumericType passedSlicePosition) {
    slicePosition = passedSlicePosition;
  }

  void setReflectX(bool passedReflectX) { reflectX = passedReflectX; }

  void apply() {
    if (inputDomain == nullptr) {
      Logger::getInstance()
          .addError("No input domain supplied to Slice.")
          .print();
    }
    if (outputDomain == nullptr) {
      Logger::getInstance()
          .addError("No output domain supplied to Slice.")
          .print();
      return;
    }

    auto materialMap = inputDomain->getMaterialMap();
    outputDomain->clear();

    for (std::size_t i = 0; i < inputDomain->getLevelSets().size(); i++) {
      viennals::Slice<NumericType> slicer(inputDomain->getLevelSets().at(i),
                                          sliceDimension, slicePosition);
      slicer.setReflectX(reflectX);
      slicer.apply();

      auto tmpLS = slicer.getSliceLevelSet();

      if (Logger::getLogLevel() >= 5) {
        auto mesh = viennals::Mesh<NumericType>::New();
        viennals::ToMesh<NumericType, 2>(tmpLS, mesh).apply();
        viennals::VTKWriter<NumericType>(mesh, "Slice_layer_" +
                                                   std::to_string(i) + ".vtp")
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

PS_PRECOMPILE_PRECISION(Slice)

} // namespace viennaps
