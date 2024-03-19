#include <psMakeHole.hpp>
#include <psProcess.hpp>
#include <psSF6O2Etching.hpp>
#include <psToSurfaceMesh.hpp>
#include <psUtils.hpp>
#include <psWriteVisualizationMesh.hpp>

#include "parameters.hpp"

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

  psLogger::setLogLevel(psLogLevel::INTERMEDIATE);
  omp_set_num_threads(16);

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

  // geometry setup
  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeHole<NumericType, D>(
      geometry, params.gridDelta /* grid delta */, params.xExtent /*x extent*/,
      params.yExtent /*y extent*/, params.holeRadius /*hole radius*/,
      params.maskHeight /* mask height*/,
      params.taperAngle /* tapering angle in degrees */, 0 /* base height */,
      false /* periodic boundary */, true /*create mask*/, psMaterial::Si)
      .apply();

  // use pre-defined model SF6O2 etching model
  auto model = psSmartPointer<psSF6O2Etching<NumericType, D>>::New(
      params.ionFlux /*ion flux*/, params.etchantFlux /*etchant flux*/,
      params.oxygenFlux /*oxygen flux*/, params.meanEnergy /*mean energy*/,
      params.sigmaEnergy /*energy sigma*/,
      params.ionExponent /*source power cosine distribution exponent*/,
      params.A_O /*oxy sputter yield*/,
      params.etchStopDepth /*max etch depth*/);

  // process setup
  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setMaxCoverageInitIterations(10);
  process.setNumberOfRaysPerPoint(params.raysPerPoint);
  process.setProcessDuration(params.processTime);

  // print initial surface
  geometry->saveSurfaceMesh("initial.vtp");

  // run the process
  process.apply();

  // print final surface
  geometry->saveSurfaceMesh("final.vtp");
}
