#include <psDomain.hpp>
#include <psMakeTrench.hpp>
#include <psProcess.hpp>
#include <psWriteVisualizationMesh.hpp>

#include <TEOSDeposition.hpp>

#include "Parameters.hpp"

int main(int argc, char **argv) {
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
  psMakeTrench<NumericType, D>(
      geometry, params.gridDelta /* grid delta */, params.xExtent /*x extent*/,
      params.yExtent /*y extent*/, params.trenchWidth /*trench width*/,
      params.trenchHeight /*trench height*/,
      params.taperAngle /* tapering angle */, 0 /*base height*/,
      false /*periodic boundary*/, false /*create mask*/,
      psMaterial::Si /*material*/)
      .apply();

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet(psMaterial::SiO2);

  // process model encompasses surface model and particle types
  auto model = psSmartPointer<TEOSDeposition<NumericType, D>>::New(
      params.stickingProbabilityP1 /*particle sticking probability*/,
      params.depositionRateP1 /*process deposition rate*/,
      params.reactionOrderP1 /*process reaction order*/,
      params.stickingProbabilityP2, params.depositionRateP2,
      params.reactionOrderP2);

  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setNumberOfRaysPerPoint(params.numRaysPerPoint);
  process.setProcessDuration(params.processTime);

  geometry->printSurface("MulitTEOS_initial.vtp");

  process.apply();

  geometry->printSurface("MulitTEOS_final.vtp");

  if constexpr (D == 2)
    psWriteVisualizationMesh<NumericType, D>(geometry, "MutliTEOS_final")
        .apply();
}