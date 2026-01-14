#include <geometries/psMakeTrench.hpp>
#include <models/psIonBeamEtching.hpp>

#include <process/psProcess.hpp>
#include <psUtil.hpp>

using namespace viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 2;

  // Parse the parameters
  util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    // Try default config file
    params.readConfigFile("config.txt");
    if (params.m.empty()) {
      std::cout << "No configuration file provided!" << std::endl;
      std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
      return 1;
    }
  }

  auto geometry = Domain<NumericType, D>::New(
      params.get("gridDelta"), params.get("xExtent"), params.get("yExtent"),
      BoundaryType::PERIODIC_BOUNDARY);
  MakeTrench<NumericType, D>(geometry, params.get("trenchWidth"),
                             params.get("trenchDepth"),
                             0.0 /*trenchTaperAngle*/, params.get("maskHeight"))
      .apply();

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet(Material::Polymer);

  IBEParameters<NumericType> ibeParams;
  ibeParams.tiltAngle = params.get("angle");
  ibeParams.exponent = params.get("exponent");
  ibeParams.thetaRMin = 0.;
  ibeParams.thetaRMax = 15.;
  ibeParams.rotatingWafer = true;

  ibeParams.meanEnergy = params.get("meanEnergy");
  ibeParams.sigmaEnergy = params.get("sigmaEnergy");
  ibeParams.thresholdEnergy = params.get("thresholdEnergy");

  ibeParams.redepositionRate = params.get("redepositionRate");
  ibeParams.planeWaferRate = params.get("planeWaferRate");

  auto model = SmartPointer<IonBeamEtching<NumericType, D>>::New(
      ibeParams, std::vector<Material>{Material::Mask});
  Vec3D<NumericType> direction{0., 0., 0.};
  direction[0] = std::sin(ibeParams.tiltAngle * M_PI / 180.);
  direction[D - 1] = -std::cos(ibeParams.tiltAngle * M_PI / 180.);
  model->setPrimaryDirection(direction);

  AdvectionParameters advectionParams;
  advectionParams.spatialScheme =
      viennals::SpatialSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER;

  Process<NumericType, D> process(geometry, model);
  process.setProcessDuration(params.get("processTime"));
  process.setParameters(advectionParams);

  geometry->saveHullMesh("initial");

  process.apply();

  geometry->saveHullMesh("final");
}
