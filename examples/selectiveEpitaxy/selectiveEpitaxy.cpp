#include <geometries/psMakePlane.hpp>
#include <models/psAnisotropicProcess.hpp>
#include <psProcess.hpp>
#include <psUtils.hpp>

#include "parameters.hpp"

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

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
  // substrate
  psMakePlane<NumericType, D>(geometry, params.gridDelta, params.xExtent,
                              params.yExtent, 0., false, psMaterial::Mask)
      .apply();
  // create fin on substrate
  {
    auto fin = psSmartPointer<lsDomain<NumericType, D>>::New(
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
        fin, psSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
        .apply();
    geometry->insertNextLevelSetAsMaterial(fin, psMaterial::Si);
    geometry->saveSurfaceMesh("fin.vtp");
  }

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet(psMaterial::SiGe);

  auto model = psSmartPointer<psAnisotropicProcess<NumericType, D>>::New(
      std::vector<std::pair<psMaterial, NumericType>>{
          {psMaterial::Si, params.epitaxyRate},
          {psMaterial::SiGe, params.epitaxyRate}});

  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setProcessDuration(params.processTime);
  process.setIntegrationScheme(
      lsIntegrationSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER);

  geometry->saveVolumeMesh("initial");

  process.apply();

  geometry->saveVolumeMesh("final");
}
