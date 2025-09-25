#include <geometries/psMakeStack.hpp>
#include <models/psFluorocarbonEtching.hpp>

#include <process/psProcess.hpp>
#include <psExtrude.hpp>

using namespace viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 2;

  // set process verbosity
  Logger::setLogLevel(LogLevel::INTERMEDIATE);

  // Parse the parameters
  util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  // set parameter units
  units::Length::setUnit(params.get<std::string>("lengthUnit"));
  units::Time::setUnit(params.get<std::string>("timeUnit"));

  // geometry setup
  auto geometry = Domain<NumericType, D>::New(
      params.get("gridDelta"), params.get("xExtent"), params.get("yExtent"));
  MakeStack<NumericType, D>(geometry, params.get<int>("numLayers"),
                            params.get("layerHeight"),
                            params.get("substrateHeight"),
                            0.0, // holeRadius
                            params.get("trenchWidth"), params.get("maskHeight"))
      .apply();

  // copy top layer for deposition
  geometry->duplicateTopLevelSet(Material::Polymer);

  // use pre-defined Fluorocarbon etching model
  auto parameters = FluorocarbonParameters<NumericType>();
  parameters.addMaterial({.density = 2.2, .id = Material::SiO2});
  parameters.addMaterial({.density = 2.,
                          .beta_e = 0.6,
                          .A_ie = 0.0361 * 2,
                          .id = Material::Polymer});
  parameters.addMaterial({.density = 2.3, .id = Material::Si3N4});
  parameters.addMaterial({.density = 5.5, .id = Material::Si});
  parameters.addMaterial({.density = 500.,
                          .beta_e = 0.1,
                          .beta_p = 0.01,
                          .Eth_sp = 20.,
                          .id = Material::Mask});

  parameters.ionFlux = params.get("ionFlux");
  parameters.etchantFlux = params.get("etchantFlux");
  parameters.polyFlux = params.get("polyFlux");
  parameters.Ions.meanEnergy = params.get("meanIonEnergy");
  parameters.Ions.sigmaEnergy = params.get("sigmaIonEnergy");
  parameters.Ions.exponent = params.get("ionExponent");

  auto model =
      SmartPointer<FluorocarbonEtching<NumericType, D>>::New(parameters);

  AdvectionParameters advectionParams;
  advectionParams.integrationScheme =
      IntegrationScheme::LOCAL_LAX_FRIEDRICHS_1ST_ORDER;
  advectionParams.timeStepRatio = 0.25;

  CoverageParameters coverageParams;
  coverageParams.maxIterations = 10;

  // process setup
  Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setProcessDuration(params.get("processTime"));
  process.setParameters(advectionParams);
  process.setParameters(coverageParams);

  // print initial surface
  geometry->saveVolumeMesh("initial");

  process.apply();

  // print final surface
  geometry->saveVolumeMesh("final");

  std::cout << "Extruding to 3D ..." << std::endl;
  auto extruded = Domain<NumericType, 3>::New();
  Vec2D<NumericType> extrudeExtent{-20., 20.};
  Extrude<NumericType>(geometry, extruded, extrudeExtent, 0,
                       {viennals::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
                        viennals::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
                        viennals::BoundaryConditionEnum::INFINITE_BOUNDARY})
      .apply();

  extruded->saveHullMesh("final_extruded");
}
