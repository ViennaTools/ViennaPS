#include <psDomain.hpp>

#include "blazedGratingsGeometry.hpp"

#include <models/psIonBeamEtching.hpp>
#include <psConstants.hpp>
#include <psProcess.hpp>
#include <psProcessParams.hpp>

using namespace viennaps;

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 2;
  omp_set_num_threads(16);

  Logger::setLogLevel(LogLevel::DEBUG);

  // Parse the parameters
  utils::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    params.readConfigFile(
        "/home/reiter/Code/ViennaPS/examples/blazedGratingsEtching/config.txt");
    // std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    // return 1;
  }

  auto geometry = generateMask<NumericType, D>(
      params.get("bumpWidth"), params.get("bumpHeight"),
      params.get<int>("numBumps"), params.get("bumpDuty"),
      params.get("gridDelta"), params.get("yExtent"));

  geometry->saveHullMesh("initial");

  AdvectionParameters<NumericType> advParams;
  advParams.integrationScheme =
      viennals::IntegrationSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER;
  advParams.timeStepRatio = 0.2;

  { // ANSGM etch
    const NumericType angle = params.get("phi1");

    IBEParameters<NumericType> ibeParams;
    ibeParams.tiltAngle = angle;

    auto model = SmartPointer<IonBeamEtching<NumericType, D>>::New();
    const NumericType angleRad = constants::degToRad(angle);
    const NumericType x = -std::sin(angleRad);
    const NumericType y = -std::cos(angleRad);
    model->setPrimaryDirection({x, y, 0});
    model->setProcessName("ANSGM_etch");

    Process<NumericType, D> process;
    process.setDomain(geometry);
    process.setProcessModel(model);
    process.setProcessDuration(params.get("ANSGM_depth"));
    process.setAdvectionParameters(advParams);

    process.apply();

    geometry->saveHullMesh("ANSGM_etch");
  }
}