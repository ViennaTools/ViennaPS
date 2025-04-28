#include <geometries/psMakePlane.hpp>
#include <models/psHBrO2Etching.hpp>
#include <psProcess.hpp>
#include <psUtil.hpp>

#include <psDomain.hpp>
#include <psGDSReader.hpp>
#include <psMaterials.hpp>

namespace ps = viennaps;
namespace ls = viennals;

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 3;

  ps::Logger::setLogLevel(ps::LogLevel::ERROR);
  omp_set_num_threads(12);

  // Parse the parameters
  ps::util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  // set parameter units
  ps::units::Length::setUnit(params.get<std::string>("lengthUnit"));
  ps::units::Time::setUnit(params.get<std::string>("timeUnit"));

  constexpr NumericType gridDelta = 0.005 * (1. + 1e-12);
  ls::BoundaryConditionEnum boundaryConds[D] = {
      ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
      ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
      ls::BoundaryConditionEnum::INFINITE_BOUNDARY};

  auto mask = ps::SmartPointer<ps::GDSGeometry<NumericType, D>>::New(
      gridDelta, boundaryConds);
  ps::GDSReader<NumericType, D>(mask, params.get<std::string>("gdsFile"))
      .apply();

  // geometry setup
  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
  auto maskLS = mask->layerToLevelSet(0, 0.0, 0.18, true);
  geometry->insertNextLevelSetAsMaterial(maskLS, ps::Material::Mask);

  ps::MakePlane<NumericType, D>(geometry, 0.0, ps::Material::Si, true).apply();

  ps::HBrO2Parameters<NumericType> modelParams;
  modelParams.ionFlux = params.get("ionFlux");
  modelParams.etchantFlux = params.get("etchantFlux");
  modelParams.oxygenFlux = params.get("oxygenFlux");
  modelParams.Ions.meanEnergy = params.get("meanEnergy");
  modelParams.Ions.sigmaEnergy = params.get("sigmaEnergy");
  modelParams.Ions.exponent = params.get("ionExponent");
  modelParams.Ions.n_l = 200;
  auto model =
      ps::SmartPointer<ps::HBrO2Etching<NumericType, D>>::New(modelParams);

  // Process setup
  ps::Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setCoverageDeltaThreshold(1e-4);
  process.setNumberOfRaysPerPoint(static_cast<int>(params.get("raysPerPoint")));
  process.setProcessDuration(params.get("processTime"));
  process.setIntegrationScheme(ps::util::convertIntegrationScheme(
      params.get<std::string>("integrationScheme")));

  // print initial surface
  geometry->saveSurfaceMesh("DRAM_Initial.vtp");

  const int numSteps = params.get("numSteps");
  for (int i = 0; i < numSteps; ++i) {
    process.apply();
    geometry->saveSurfaceMesh("DRAM_Etched_" + std::to_string(i + 1) + ".vtp");
  }

  geometry->saveHullMesh("DRAM_Final");

  return 0;
}
