#include <geometries/psMakeFin.hpp>
#include <geometries/psMakeHole.hpp>
#include <geometries/psMakePlane.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psSelectiveEpitaxy.hpp>
#include <psProcess.hpp>

namespace ps = viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

  // Parse the parameters
  ps::util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  auto geometry = ps::Domain<NumericType, D>::New(
      params.get("gridDelta"), params.get("xExtent"), params.get("yExtent"));

  // substrate
  ps::MakeFin<NumericType, D>(geometry, params.get("finWidth"),
                              params.get("finHeight"))
      .apply();

  // oxide layer
  ps::MakePlane<NumericType, D>(geometry, params.get("oxideHeight"),
                                ps::Material::SiO2, true)
      .apply();

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet(ps::Material::SiGe);

  auto model = ps::SmartPointer<ps::SelectiveEpitaxy<NumericType, D>>::New(
      std::vector<std::pair<ps::Material, NumericType>>{
          {ps::Material::Si, params.get("epitaxyRate")},
          {ps::Material::SiGe, params.get("epitaxyRate")}},
      params.get("R111"), params.get("R100"));

  ps::AdvectionParameters<NumericType> advectionParams;
  advectionParams.integrationScheme =
      viennals::IntegrationSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER;
  // advectionParams.velocityOutput = true;
  lsInternal::StencilLocalLaxFriedrichsScalar<NumericType, D,
                                              1>::setMaxDissipation(1000);

  ps::Process<NumericType, D> process(geometry, model,
                                      params.get("processTime"));
  process.setAdvectionParameters(advectionParams);

  geometry->saveVolumeMesh("initial_fin");

  process.apply();

  geometry->saveVolumeMesh("final_fin");

  geometry->clear();
  ps::MakeTrench(geometry, params.get("finWidth"), 0., 0.,
                 params.get("finHeight"), 0., false, ps::Material::Si,
                 ps::Material::SiO2)
      .apply();
  geometry->duplicateTopLevelSet(ps::Material::SiGe);

  geometry->saveVolumeMesh("initial_trench");

  process.apply();

  geometry->saveVolumeMesh("final_trench");
}
