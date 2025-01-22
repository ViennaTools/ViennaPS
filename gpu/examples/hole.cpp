#include <geometries/psMakeHole.hpp>

#include <gpu/vcContext.hpp>

#include <pscuProcess.hpp>
#include <pscuSF6O2Etching.hpp>

#include <psUtils.hpp>

using namespace viennaps;

int main(int argc, char **argv) {

  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = float;

  Logger::setLogLevel(LogLevel::DEBUG);

  // Parse the parameters
  utils::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  Context context;
  CreateContext(context);

  // geometry setup
  auto geometry = SmartPointer<Domain<NumericType, D>>::New();
  MakeHole<NumericType, D>(
      geometry, params.get("gridDelta"), params.get("xExtent"),
      params.get("yExtent"), params.get("holeRadius"), params.get("maskHeight"),
      params.get("taperAngle"), 0, false, true, Material::Si)
      .apply();

  // use pre-defined model SF6O2 etching model
  SF6O2Parameters<NumericType> modelParams;
  modelParams.ionFlux = params.get("ionFlux");
  modelParams.etchantFlux = params.get("etchantFlux");
  modelParams.oxygenFlux = params.get("oxygenFlux");
  modelParams.Ions.minEnergy = params.get("meanEnergy");
  modelParams.Ions.deltaEnergy = params.get("sigmaEnergy");
  modelParams.Ions.exponent = params.get("ionExponent");
  modelParams.Passivation.A_ie = params.get("A_O");
  modelParams.print();

  auto model =
      SmartPointer<gpu::SF6O2Etching<NumericType, D>>::New(modelParams);

  gpu::Process<NumericType, D> process(context, geometry, model);
  process.setNumberOfRaysPerPoint(3000);
  process.setProcessParams(modelParams);
  process.setMaxCoverageInitIterations(10);
  process.setProcessDuration(100.);
  process.setIntegrationScheme(
      viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_2ND_ORDER);
  process.apply();

  geometry->saveSurfaceMesh("final.vtp");
}