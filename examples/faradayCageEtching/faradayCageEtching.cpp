#include <geometries/psMakeFin.hpp>
#include <models/psFaradayCageEtching.hpp>

#include <psProcess.hpp>
#include <psUtils.hpp>

namespace ps = viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

  ps::Logger::setLogLevel(ps::LogLevel::INTERMEDIATE);
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
  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New(
      params.get("gridDelta"), params.get("xExtent"), params.get("yExtent"),
      ps::BoundaryType::PERIODIC_BOUNDARY);
  ps::MakeFin<NumericType, D>(geometry, params.get("finWidth"),
                              0.0 /*finHeight*/, 0.0 /*finTaperAngle*/,
                              params.get("maskHeight"))
      .apply();

  std::vector<ps::Material> maskMaterials = {ps::Material::Mask};
  ps::FaradayCageParameters<NumericType> cageParams;
  cageParams.ibeParams.tiltAngle = params.get("tiltAngle");
  cageParams.cageAngle = params.get("cageAngle");
  auto model = ps::SmartPointer<ps::FaradayCageEtching<NumericType, D>>::New(
      maskMaterials, cageParams);

  // process setup
  ps::Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setNumberOfRaysPerPoint(params.get("raysPerPoint"));
  process.setProcessDuration(params.get("etchTime"));
  process.setIntegrationScheme(
      ps::IntegrationScheme::LOCAL_LAX_FRIEDRICHS_1ST_ORDER);

  // print initial surface
  geometry->saveHullMesh("initial.vtp");

  // run the process
  process.apply();

  // print final surface
  geometry->saveHullMesh("final");
}
