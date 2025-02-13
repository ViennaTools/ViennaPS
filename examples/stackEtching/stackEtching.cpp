#include <geometries/psMakeStack.hpp>
#include <models/psFluorocarbonEtching.hpp>

#include <psExtrude.hpp>
#include <psProcess.hpp>

namespace ps = viennaps;
namespace ls = viennals;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 2;

  // set process verbosity
  ps::Logger::setLogLevel(ps::LogLevel::INTERMEDIATE);

  // Parse the parameters
  ps::utils::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  // geometry setup
  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
  ps::MakeStack<NumericType, D>(
      geometry, params.get("gridDelta"), params.get("xExtent"),
      params.get("yExtent"), params.get<int>("numLayers"),
      params.get("layerHeight"), params.get("substrateHeight"), 0.0,
      params.get("trenchWidth"), params.get("maskHeight"), false)
      .apply();

  // copy top layer for deposition
  geometry->duplicateTopLevelSet(ps::Material::Polymer);

  // use pre-defined model Fluorocarbon etching model
  auto model = ps::SmartPointer<ps::FluorocarbonEtching<NumericType, D>>::New(
      params.get("ionFlux"), params.get("etchantFlux"), params.get("polyFlux"),
      params.get("meanIonEnergy"), params.get("sigmaIonEnergy"),
      params.get("ionExponent"));

  // process setup
  ps::Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setProcessDuration(params.get("processTime"));
  process.setMaxCoverageInitIterations(10);
  process.setTimeStepRatio(0.25);

  // print initial surface
  geometry->saveVolumeMesh("initial");

  process.apply();

  // print final surface
  geometry->saveVolumeMesh("final");

  std::cout << "Extruding to 3D ..." << std::endl;
  auto extruded = ps::SmartPointer<ps::Domain<NumericType, 3>>::New();
  std::array<NumericType, 2> extrudeExtent = {-20., 20.};
  ps::Extrude<NumericType>(geometry, extruded, extrudeExtent, 0,
                           {ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
                            ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
                            ls::BoundaryConditionEnum::INFINITE_BOUNDARY})
      .apply();

  extruded->saveVolumeMesh("final_extruded");
}
