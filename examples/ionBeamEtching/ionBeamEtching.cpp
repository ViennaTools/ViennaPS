#include <geometries/psMakeTrench.hpp>
#include <models/psIonBeamEtching.hpp>

#include <psProcess.hpp>
#include <psUtil.hpp>

using namespace viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 2;
  Logger::setLogLevel(LogLevel::INTERMEDIATE);

  // Parse the parameters
  util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  auto geometry = SmartPointer<Domain<NumericType, D>>::New(
      params.get("gridDelta"), params.get("xExtent"), params.get("yExtent"));
  MakeTrench<NumericType, D>(geometry, params.get("trenchWidth"),
                             params.get("trenchDepth"),
                             0.0 /*trenchTaperAngle*/, params.get("maskHeight"))
      .apply();

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet(Material::Polymer);

  IBEParameters<NumericType> ibeParams;
  ibeParams.tiltAngle = params.get("angle");
  ibeParams.exponent = params.get("exponent");

  ibeParams.meanEnergy = params.get("meanEnergy");
  ibeParams.sigmaEnergy = params.get("sigmaEnergy");
  ibeParams.thresholdEnergy = params.get("thresholdEnergy");

  ibeParams.redepositionRate = params.get("redepositionRate");
  ibeParams.planeWaferRate = params.get("planeWaferRate");

  auto model = SmartPointer<IonBeamEtching<NumericType, D>>::New(
      std::vector<Material>{Material::Mask}, ibeParams);
  Vec3D<NumericType> direction{0., 0., 0.};
  direction[D - 1] = -std::cos(ibeParams.tiltAngle * M_PI / 180.);
  direction[D - 2] = std::sin(ibeParams.tiltAngle * M_PI / 180.);
  model->setPrimaryDirection(direction);

  Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setProcessDuration(params.get("processTime"));
  process.setIntegrationScheme(
      viennals::IntegrationSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER);

  geometry->saveHullMesh("initial");

  process.apply();

  geometry->saveHullMesh("final");
}
