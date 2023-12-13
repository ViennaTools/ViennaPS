#include <geometries/psMakePlane.hpp>
#include <psProcess.hpp>
#include <psSelectiveEpitaxy.hpp>
#include <psUtils.hpp>
#include <psWriteVisualizationMesh.hpp>

#include "parameters.hpp"

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 2;

  // Parse the parameters
  Parameters<NumericType> params;
  if (argc > 1) {
    auto config = psUtils::readConfigFile(argv[1]);
    if (config.empty()) {
      std::cerr << "Empty config provided" << std::endl;
      return -1;
    }
    params.fromMap(config);
  }

  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakePlane<NumericType, D>(geometry, params.gridDelta, params.xExtent,
                              params.yExtent, 0., false, psMaterial::Si)
      .apply();
  {
    auto box = psSmartPointer<lsDomain<NumericType, D>>::New(
        geometry->getLevelSets()->back()->getGrid());
    NumericType minPoint[3] = {-params.finWidth / 2., -params.finLength / 2.,
                               -params.gridDelta};
    NumericType maxPoint[3] = {params.finWidth / 2., params.finLength / 2.,
                               params.finHeight};
    if constexpr (D == 2) {
      minPoint[1] = -params.gridDelta;
      maxPoint[1] = params.finHeight;
    }
    lsMakeGeometry<NumericType, D>(
        box, psSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
        .apply();
    geometry->applyBooleanOperation(box, lsBooleanOperationEnum::UNION);
  }

  psMakePlane<NumericType, D>(geometry, params.maskHeight, psMaterial::Mask)
      .apply();

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet(psMaterial::SiO2);

  auto model = psSmartPointer<psSelectiveEpitaxy<NumericType, D>>::New(
      std::vector<psMaterial>{psMaterial::Si, psMaterial::SiO2});

  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setProcessDuration(params.processTime);
  process.setIntegrationScheme(
      lsIntegrationSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER);

  geometry->printSurface("initial.vtp");
  psWriteVisualizationMesh<NumericType, D>(geometry, "initial").apply();

  process.apply();

  geometry->printSurface("final.vtp");
  psWriteVisualizationMesh<NumericType, D>(geometry, "final").apply();
}
