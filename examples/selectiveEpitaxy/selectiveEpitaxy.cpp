#include <geometries/psMakePlane.hpp>
#include <models/psAnisotropicProcess.hpp>
#include <psProcess.hpp>
#include <psUtils.hpp>

namespace ps = viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

  // Parse the parameters
  ps::utils::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
  // substrate
  ps::MakePlane<NumericType, D>(geometry, params.get("gridDelta"),
                                params.get("xExtent"), params.get("yExtent"),
                                0., false, ps::Material::Mask)
      .apply();
  // create fin on substrate
  {
    auto fin =
        lsSmartPointer<lsDomain<NumericType, D>>::New(geometry->getGrid());
    NumericType minPoint[3] = {-params.get("finWidth") / 2.,
                               -params.get("finLength") / 2.,
                               -params.get("gridDelta")};
    NumericType maxPoint[3] = {params.get("finWidth") / 2.,
                               params.get("finLength") / 2.,
                               params.get("finHeight")};
    if constexpr (D == 2) {
      minPoint[1] = -params.get("gridDelta");
      maxPoint[1] = params.get("finHeight");
    }
    lsMakeGeometry<NumericType, D>(
        fin, lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
        .apply();
    geometry->insertNextLevelSetAsMaterial(fin, ps::Material::Si);
    geometry->saveSurfaceMesh("fin.vtp");
  }

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet(ps::Material::SiGe);

  auto model = ps::SmartPointer<ps::AnisotropicProcess<NumericType, D>>::New(
      std::vector<std::pair<ps::Material, NumericType>>{
          {ps::Material::Si, params.get("epitaxyRate")},
          {ps::Material::SiGe, params.get("epitaxyRate")}});

  ps::Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setProcessDuration(params.get("processTime"));
  process.setIntegrationScheme(
      lsIntegrationSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER);

  geometry->saveVolumeMesh("initial");

  process.apply();

  geometry->saveVolumeMesh("final");
}
