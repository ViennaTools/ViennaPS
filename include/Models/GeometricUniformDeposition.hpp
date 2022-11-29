#pragma once

#include <lsGeometricAdvect.hpp>
#include <lsGeometricAdvectDistributions.hpp>

#include <psGeometricModel.hpp>
#include <psProcessModel.hpp>
#include <psSmartPointer.hpp>

// Simple geometric model that implements a
template <typename NumericType, int D>
class GeometricUniformDepositionModel
    : public psGeometricModel<NumericType, D> {
  NumericType layerThickness;

  using psGeometricModel<NumericType, D>::domain;

public:
  GeometricUniformDepositionModel(NumericType passedLayerThickness)
      : layerThickness(passedLayerThickness) {}

  void apply() {
    auto dist = psSmartPointer<lsSphereDistribution<NumericType, D>>::New(
        layerThickness, domain->getGrid().getGridDelta());
    // TODO: is back or front of vector of levelsets the top levelset?
    lsGeometricAdvect<NumericType, D>(domain->getLevelSets()->back(), dist)
        .apply();
  }
};

template <typename NumericType, int D> class GeometricUniformDeposition {
  psSmartPointer<psProcessModel<NumericType, D>> processModel = nullptr;

public:
  GeometricUniformDeposition(const NumericType layerThickness = 1.) {
    processModel = psSmartPointer<psProcessModel<NumericType, D>>::New();

    auto geomModel =
        psSmartPointer<GeometricUniformDepositionModel<NumericType, D>>::New(
            layerThickness);

    processModel->setGeometricModel(geomModel);
    processModel->setProcessName("GeometricUniformDeposition");
  }

  psSmartPointer<psProcessModel<NumericType, D>> getProcessModel() {
    return processModel;
  }
};
