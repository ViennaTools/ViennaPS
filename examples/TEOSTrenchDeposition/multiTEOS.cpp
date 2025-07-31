#include <geometries/psMakeTrench.hpp>
#include <models/psTEOSDeposition.hpp>

#include <psDomain.hpp>
#include <psProcess.hpp>
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
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  auto geometry = ps::Domain<NumericType, D>::New(
      params.get("gridDelta"), params.get("xExtent"), params.get("yExtent"));
  ps::MakeTrench<NumericType, D>(geometry, params.get("trenchWidth"),
                                 params.get("trenchHeight"),
                                 params.get("taperAngle"))
      .apply();

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet(ps::Material::SiO2);

  // process model with surface model and particle types
  auto model = ps::SmartPointer<ps::TEOSDeposition<NumericType, D>>::New(
      params.get("stickingProbabilityP1"), params.get("depositionRateP1"),
      params.get("reactionOrderP1"), params.get("stickingProbabilityP2"),
      params.get("depositionRateP2"), params.get("reactionOrderP2"));

  ps::Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setNumberOfRaysPerPoint(params.get<unsigned>("numRaysPerPoint"));
  process.setProcessDuration(params.get("processTime"));

  geometry->saveVolumeMesh("MulitTEOS_initial");

  process.apply();

  geometry->saveVolumeMesh("MulitTEOS_final");
}
