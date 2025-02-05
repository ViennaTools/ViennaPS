#include <geometries/psMakeHole.hpp>
#include <models/psSF6O2Etching.hpp>

#include <psProcess.hpp>
#include <psUtils.hpp>

namespace ps = viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 2;

  ps::Logger::setLogLevel(ps::LogLevel::DEBUG);
  omp_set_num_threads(16);

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
  ps::MakeHole<NumericType, D>(
      geometry, params.get("gridDelta"), params.get("xExtent"),
      params.get("yExtent"), params.get("holeRadius"), params.get("maskHeight"),
      params.get("taperAngle"), 0 /* base height */,
      false /* periodic boundary */, true /*create mask*/, ps::Material::Si)
      .apply();

  // set parameter units
  ps::units::Length::setUnit(params.get<std::string>("lengthUnit"));
  ps::units::Time::setUnit(params.get<std::string>("timeUnit"));

  // use pre-defined model SF6O2 etching model
  ps::SF6O2Parameters<NumericType> modelParams;
  modelParams.ionFlux = params.get("ionFlux");
  modelParams.etchantFlux = params.get("etchantFlux");
  modelParams.oxygenFlux = params.get("oxygenFlux");
  modelParams.Ions.meanEnergy = params.get("meanEnergy");
  modelParams.Ions.sigmaEnergy = params.get("sigmaEnergy");
  modelParams.Ions.exponent = params.get("ionExponent");
  modelParams.Passivation.A_ie = params.get("A_O");
  modelParams.Si.A_ie = params.get("A_Si");
  modelParams.etchStopDepth = params.get("etchStopDepth");
  auto model =
      ps::SmartPointer<ps::SF6O2Etching<NumericType, D>>::New(modelParams);

  // process setup
  ps::Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setMaxCoverageInitIterations(50);
  process.setNumberOfRaysPerPoint(params.get("raysPerPoint"));
  process.setProcessDuration(params.get("processTime"));
  process.setIntegrationScheme(
      params.get<viennals::IntegrationSchemeEnum>("integrationScheme"));

  // print initial surface
  geometry->saveSurfaceMesh("initial.vtp");

  // run the process
  process.apply();

  // print final surface
  geometry->saveSurfaceMesh(params.get<std::string>("outputFile"));
}
