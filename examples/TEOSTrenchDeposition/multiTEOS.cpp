#include <geometries/psMakeTrench.hpp>
#include <models/psTEOSDeposition.hpp>

#include <psDomain.hpp>
#include <psProcess.hpp>
#include <psUtils.hpp>

namespace ps = viennaps;

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 2;

  // Parse the parameters
  ps::utils::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
  ps::MakeTrench<NumericType, D>(
      geometry, params.get("gridDelta") /* grid delta */,
      params.get("xExtent") /*x extent*/, params.get("yExtent") /*y extent*/,
      params.get("trenchWidth") /*trench width*/,
      params.get("trenchHeight") /*trench height*/,
      params.get("taperAngle") /* tapering angle */, 0 /*base height*/,
      false /*periodic boundary*/, false /*create mask*/,
      ps::Material::Si /*material*/)
      .apply();

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet(ps::Material::SiO2);

  // process model encompasses surface model and particle types
  auto model = ps::SmartPointer<ps::TEOSDeposition<NumericType, D>>::New(
      params.get("stickingProbabilityP1") /*particle sticking probability*/,
      params.get("depositionRateP1") /*process deposition rate*/,
      params.get("reactionOrderP1") /*process reaction order*/,
      params.get("stickingProbabilityP2"), params.get("depositionRateP2"),
      params.get("reactionOrderP2"));

  ps::Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setNumberOfRaysPerPoint(params.get("numRaysPerPoint"));
  process.setProcessDuration(params.get("processTime"));

  geometry->saveSurfaceMesh("MulitTEOS_initial.vtp");

  process.apply();

  geometry->saveSurfaceMesh("MulitTEOS_final.vtp");
  geometry->saveVolumeMesh("MulitTEOS_final");
}
