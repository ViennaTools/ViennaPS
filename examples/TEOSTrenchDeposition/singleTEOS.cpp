#include <geometries/psMakeTrench.hpp>
#include <models/psTEOSDeposition.hpp>

#include <process/psProcess.hpp>
#include <psDomain.hpp>
#include <psUtil.hpp>

namespace ps = viennaps;

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 2;

  // Parse the parameters
  ps::util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    // Try default config file
    params.readConfigFile("singleTEOS_config.txt");
    if (params.m.empty()) {
      std::cout << "No configuration file provided!" << std::endl;
      std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
      return 1;
    }
  }

  auto geometry = ps::Domain<NumericType, D>::New(
      params.get("gridDelta"), params.get("xExtent"), params.get("yExtent"));
  ps::MakeTrench<NumericType, D>(geometry, params.get("trenchWidth"),
                                 params.get("trenchHeight"),
                                 params.get("taperAngle"))
      .apply();

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet(ps::Material::SiO2);

  // process model encompasses surface model and particle types
  auto model = ps::SmartPointer<ps::TEOSDeposition<NumericType, D>>::New(
      params.get("stickingProbabilityP1"), params.get("depositionRateP1"),
      params.get("reactionOrderP1"));

  ps::RayTracingParameters rayParams;
  rayParams.raysPerPoint = params.get<unsigned>("numRaysPerPoint");

  ps::Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setParameters(rayParams);
  process.setProcessDuration(params.get("processTime"));

  geometry->saveVolumeMesh("SingleTEOS_initial");

  process.apply();

  geometry->saveVolumeMesh("SingleTEOS_final");
}
