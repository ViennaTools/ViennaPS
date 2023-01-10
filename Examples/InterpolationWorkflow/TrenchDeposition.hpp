#pragma once

#include <Geometries/psMakeTrench.hpp>
#include <SimpleDeposition.hpp>
#include <psCSVWriter.hpp>
#include <psProcess.hpp>
#include <psToSurfaceMesh.hpp>
#include <psUtils.hpp>

#include "AdvectionCallback.hpp"
#include "Parameters.hpp"

template <typename NumericType, int D, int DataDimensions = 0>
void executeProcess(
    psSmartPointer<psDomain<NumericType, D>> geometry,
    const Parameters<NumericType> &params,
    psSmartPointer<AdvectionCallback<NumericType, D, DataDimensions>>
        advectionCallback = nullptr) {
  // copy top layer to capture deposition
  auto depoLayer = psSmartPointer<lsDomain<NumericType, D>>::New(
      geometry->getLevelSets()->back());
  geometry->insertNextLevelSet(depoLayer);

  auto processModel =
      SimpleDeposition<NumericType, D>(
          params.stickingProbability /*particle sticking probability*/,
          params.sourcePower /*particel source power*/)
          .getProcessModel();

  if (advectionCallback)
    processModel->setAdvectionCallback(advectionCallback);

  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(processModel);
  process.setNumberOfRaysPerPoint(1000);
  process.setProcessDuration(params.processTime / params.stickingProbability);

  process.apply();
}