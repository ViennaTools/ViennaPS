#include <geometries/psMakeFin.hpp>
#include <models/psFaradayCageEtching.hpp>

#include <psProcess.hpp>
#include <psUtils.hpp>

namespace ps = viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

  ps::Logger::setLogLevel(ps::LogLevel::INFO);
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
  ps::MakeFin<NumericType, D>(geometry, params.get("gridDelta"),
                              params.get("xExtent"), params.get("yExtent"),
                              params.get("finWidth"), params.get("maskHeight"),
                              0.0, 0.0, true, true, ps::Material::Si)
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
  process.setMaxCoverageInitIterations(10);
  process.setNumberOfRaysPerPoint(params.get("raysPerPoint"));
  process.setProcessDuration(params.get("etchTime"));

  // print initial surface
  geometry->saveSurfaceMesh("initial.vtp");

  // run the process
  process.apply();

  // print final surface
  geometry->saveSurfaceMesh("final.vtp");
}
