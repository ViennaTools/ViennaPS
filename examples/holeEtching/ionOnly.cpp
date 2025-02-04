#include <geometries/psMakeHole.hpp>
#include <models/ion_only.hpp>

#include <psConstants.hpp>
#include <psProcess.hpp>
#include <psUtils.hpp>

namespace ps = viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 2;

  ps::Logger::setLogLevel(ps::LogLevel::INTERMEDIATE);
  omp_set_num_threads(16);

  // geometry setup
  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
  ps::MakeHole<NumericType, D>(geometry, 0.02, 2.0, 2.0, 0.175, 1.2, 1.193, 0,
                               false /* periodic boundary */,
                               true /*create mask*/, ps::Material::Si)
      .apply();

  // use pre-defined model SF6O2 etching model
  ps::SF6O2Parameters<NumericType> modelParams;
  modelParams.Ions.meanEnergy = 100.0;
  modelParams.Ions.sigmaEnergy = 10.0;
  modelParams.Ions.exponent = 500;
  modelParams.Ions.inflectAngle = ps::constants::degToRad(89.);
  modelParams.Ions.minAngle = ps::constants::degToRad(85.);
  modelParams.Ions.n_l = 10.;
  modelParams.Si.A_ie = 7.0;

  modelParams.Si.rho = 0.1;
  modelParams.Mask.rho = 0.05;

  auto model =
      ps::SmartPointer<ps::IonOnlyEtching<NumericType, D>>::New(modelParams);

  // process setup
  ps::Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setProcessDuration(10.0 / 60.0);
  process.setTimeStepRatio(0.25);
  process.setIntegrationScheme(
      viennals::IntegrationSchemeEnum::LAX_FRIEDRICHS_1ST_ORDER);

  // print initial surface
  geometry->saveSurfaceMesh("initial.vtp");

  // run the process
  process.apply();

  // print final surface
  geometry->saveSurfaceMesh("ionOnly");
}
