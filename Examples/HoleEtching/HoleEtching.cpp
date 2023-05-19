#include <SF6O2Etching.hpp>
#include <psMakeHole.hpp>
#include <psProcess.hpp>
#include <psToSurfaceMesh.hpp>
#include <psUtils.hpp>

#include "Parameters.hpp"

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

  // geometry setup
  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeHole<NumericType, D>(
      geometry, params.gridDelta /* grid delta */, params.xExtent /*x extent*/,
      params.yExtent /*y extent*/, params.holeRadius /*hole radius*/,
      params.maskHeight /* mask height*/,
      params.taperAngle /* tapering angle in degrees */, 0 /* base height */,
      false /* periodic boundary */, true /*create mask*/, psMaterial::Si)
      .apply();

  // use pre-defined model SF6O2 etching
  auto model = psSmartPointer<SF6O2Etching<NumericType, D>>::New(
      params.totalIonFlux /*ion flux*/,
      params.totalEtchantFlux /*etchant flux*/,
      params.totalOxygenFlux /*oxygen flux*/, params.rfBias /*rf bias*/,
      params.A_O /*oxy sputter yield*/);

  // process setup
  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setMaxCoverageInitIterations(10);
  process.setNumberOfRaysPerPoint(params.raysPerPoint);
  process.setProcessDuration(params.processTime);

  // print initial surface
  geometry->printSurface("initial.vtp");

  // run the process
  process.apply();

  // write collected particle meta data (ion energy distribution) to a file
  process.writeParticleDataLogs("ionEnergyDistribution.txt");

  // print final surface
  geometry->printSurface("final.vtp");
}
