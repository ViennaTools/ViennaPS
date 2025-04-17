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
  omp_set_num_threads(16);

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

  constexpr NumericType gridDelta = 0.005;

  ls::BoundaryConditionEnum boundaryConds[D] = {
      ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
      ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
      ls::BoundaryConditionEnum::INFINITE_BOUNDARY};

  auto mask = ps::SmartPointer<ps::GDSGeometry<NumericType, D>>::New(
      gridDelta, boundaryConds);
  ps::GDSReader<NumericType, D>(mask, params.get<std::string>("gdsFile"))
      .apply();

  // geometry setup
  auto bounds = mask->getBounds();
  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();

  // substrate plane
  NumericType origin[D] = {0., 0., 0.};
  NumericType normal[D] = {0., 0., 1.};
  auto plane = ps::SmartPointer<ls::Domain<NumericType, D>>::New(
      bounds, boundaryConds, gridDelta);
  ls::MakeGeometry<NumericType, D>(
      plane, ps::SmartPointer<ls::Plane<NumericType, D>>::New(origin, normal))
      .apply();
  geometry->insertNextLevelSetAsMaterial(plane, ps::Material::Si);

  auto maskLS = mask->layerToLevelSet(0, 0.0, 0.18, true);
  geometry->insertNextLevelSetAsMaterial(maskLS, ps::Material::Mask);

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
  process.setMaxCoverageInitIterations(10);
  process.setNumberOfRaysPerPoint(static_cast<int>(params.get("raysPerPoint")));
  process.setProcessDuration(params.get("processTime"));
  process.setIntegrationScheme(ps::util::convertIntegrationScheme(
      params.get<std::string>("integrationScheme")));

  // print initial surface
  geometry->saveSurfaceMesh("initial.vtp");

  for (int i = 1; i < 101; ++i) {
    process.apply();
    geometry->saveSurfaceMesh("etched_" + std::to_string(i) + ".vtp", true);
  }

  geometry->saveVolumeMesh("Geometry");

  return 0;
}
