#include <geometries/psMakePlane.hpp>
#include <models/psDirectionalProcess.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psSelectiveEpitaxy.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>
#include <psPlanarize.hpp>
#include <psReader.hpp>
#include <psWriter.hpp>

#include <lsReader.hpp>

using namespace viennaps;

constexpr int D = 3;
using NumericType = double;
using LevelSetType = SmartPointer<viennals::Domain<NumericType, D>>;
using DomainType = SmartPointer<Domain<NumericType, D>>;
using IsotropicProcessGeometric = SmartPointer<SphereDistribution<double, D>>;
constexpr bool volumeOutput = false;

void writeVolume(DomainType domain) {
  if (!volumeOutput)
    return;
  static int volumeNum = 0;
  std::cout << "Writing volume mesh ..." << std::flush;
  domain->saveVolumeMesh("FinFET_" + std::to_string(volumeNum), 0.05);
  std::cout << " done" << std::endl;
  ++volumeNum;
}

void writeSurface(DomainType domain) {
  static int outputNum = 0;
  domain->saveSurfaceMesh("FinFET_" + std::to_string(outputNum) + ".vtp", true);
  ++outputNum;
}

int main() {

  Logger::setLogLevel(LogLevel::ERROR);

  BoundaryType boundaryConds[D] = {BoundaryType::REFLECTIVE_BOUNDARY,
                                   BoundaryType::REFLECTIVE_BOUNDARY,
                                   BoundaryType::INFINITE_BOUNDARY};
  double bounds[2 * D] = {0, 90, 0, 100, 0, 70}; // in nanometers
  constexpr NumericType gridDelta = 0.79;
  auto domain = DomainType::New(bounds, boundaryConds, gridDelta);

  // Initialise domain with a single silicon plane (at z=70 because it is 70
  // nm high)
  MakePlane<NumericType, D>(domain, 70.0, Material::Si).apply();
  writeSurface(domain);

  // Add double patterning mask
  {
    auto ls = LevelSetType::New(domain->getGrid());
    VectorType<NumericType, D> min{30, -10, 69.9};
    VectorType<NumericType, D> max{60, 110, 90};
    viennals::MakeGeometry<NumericType, D>(
        ls, viennals::Box<NumericType, D>::New(min, max))
        .apply();

    domain->insertNextLevelSetAsMaterial(ls, Material::Mask);
  }
  writeSurface(domain);

  // Double patterning processes
  { // DP-Depo
    std::cout << "DP-Depo ..." << std::flush;
    const NumericType thickness = 4; // nm
    domain->duplicateTopLevelSet(Material::Metal);
    auto dist = IsotropicProcessGeometric::New(thickness, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  { // DP-Patterning
    std::cout << "DP-Patterning ..." << std::flush;
    const NumericType etchDepth = 6; // nm
    auto dist = SmartPointer<BoxDistribution<double, D>>::New(
        std::array<NumericType, D>{-gridDelta, -gridDelta, -etchDepth},
        gridDelta, domain->getLevelSets().front());
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  // Remove mask with boolean operation
  domain->removeMaterial(Material::Mask);
  writeSurface(domain);
  writeVolume(domain);

  // pattern si
  {
    std::cout << "Si-Patterning ..." << std::flush;
    const NumericType etchDepth = 90.; // nm
    Vec3D<NumericType> direction = {0, 0, 1};
    auto model = SmartPointer<DirectionalProcess<NumericType, D>>::New(
        direction, 1.1, 0.1, Material::Metal, false);
    Process<NumericType, D>(domain, model, etchDepth).apply();
    std::cout << " done" << std::endl;
  }
  writeVolume(domain);
  writeSurface(domain);

  // Remove DP mask (metal)
  domain->removeTopLevelSet();
  writeSurface(domain);

  // deposit STI material
  {
    std::cout << "STI Deposition ..." << std::flush;
    const NumericType thickness = 35; // nm
    domain->duplicateTopLevelSet(Material::SiO2);
    auto dist = IsotropicProcessGeometric::New(thickness, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  // CMP at 80
  Planarize<NumericType, D>(domain, 80.0).apply();
  writeVolume(domain);

  // pattern STI material
  {
    std::cout << "STI Patterning ..." << std::flush;
    auto dist = IsotropicProcessGeometric::New(-35, gridDelta,
                                               domain->getLevelSets().front());
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);
  writeVolume(domain);

  // deposit gate material
  {
    std::cout << "Gate Deposition HfO2 ..." << std::flush;
    const NumericType thickness = 2;              // nm
    domain->duplicateTopLevelSet(Material::HfO2); // add layer
    auto dist = IsotropicProcessGeometric::New(thickness, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  {
    std::cout << "Gate Deposition PolySi ..." << std::flush;
    const NumericType thickness = 104;              // nm
    domain->duplicateTopLevelSet(Material::PolySi); // add layer
    auto dist = IsotropicProcessGeometric::New(thickness, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  // CMP at 150
  Planarize<NumericType, D>(domain, 150.0).apply();

  // dummy gate mask addition
  {
    auto ls = LevelSetType::New(domain->getGrid());
    VectorType<NumericType, D> min{-10, 30, 145};
    VectorType<NumericType, D> max{100, 70, 175};
    viennals::MakeGeometry<NumericType, D>(
        ls, SmartPointer<viennals::Box<NumericType, D>>::New(min, max))
        .apply();

    domain->insertNextLevelSetAsMaterial(ls, Material::Mask);
  }
  writeSurface(domain);

  // gate patterning
  {
    std::cout << "Dummy Gate Patterning ..." << std::flush;
    Vec3D<NumericType> direction = {0, 0, 1};
    std::vector<Material> masks = {Material::Mask, Material::Si,
                                   Material::SiO2};
    auto model = SmartPointer<DirectionalProcess<NumericType, D>>::New(
        direction, 1.0, 0., masks, false);
    Process<NumericType, D>(domain, model, 110.).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  // Remove mask
  domain->removeTopLevelSet();
  writeSurface(domain);
  writeVolume(domain);

  // Spacer Deposition and Etch
  { // spacer depo
    std::cout << "Spacer Deposition ..." << std::flush;
    const NumericType thickness = 10;              // nm
    domain->duplicateTopLevelSet(Material::Si3N4); // add layer
    auto dist = IsotropicProcessGeometric::New(thickness, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  { // spacer etch
    std::cout << "Spacer Etch ..." << std::flush;
    auto ls = domain->getLevelSets()[domain->getLevelSets().size() - 2];
    auto dist = SmartPointer<BoxDistribution<double, D>>::New(
        std::array<NumericType, D>{-gridDelta, -gridDelta, -50}, gridDelta, ls);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);
  writeVolume(domain);

  // isotropic etch (fin-release)
  {
    std::cout << "Fin-Release ..." << std::flush;
    std::vector<Material> masks = {Material::PolySi, Material::SiO2,
                                   Material::Si3N4};
    auto model =
        SmartPointer<IsotropicProcess<NumericType, D>>::New(-1., masks);
    AdvectionParameters advectionParams;
    advectionParams.integrationScheme =
        viennals::IntegrationSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER;
    Process<NumericType, D> process(domain, model, 5.);
    process.setAdvectionParameters(advectionParams);
    process.apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);
  writeVolume(domain);

  // source/drain epitaxy
  {
    std::cout << "S/D Epitaxy ..." << std::flush;
    domain->duplicateTopLevelSet(Material::SiGe);
    Logger::setLogLevel(LogLevel::INFO);
    AdvectionParameters advectionParams;
    advectionParams.integrationScheme =
        viennals::IntegrationSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER;
    lsInternal::StencilLocalLaxFriedrichsScalar<NumericType, D,
                                                1>::setMaxDissipation(100);

    std::vector<std::pair<Material, NumericType>> material = {
        {Material::Si, 1.}, {Material::SiGe, 1.}};
    auto model = SmartPointer<SelectiveEpitaxy<NumericType, D>>::New(material);
    Process<NumericType, D> proc(domain, model, 14.);
    proc.setAdvectionParameters(advectionParams);
    proc.apply();

    std::cout << " done" << std::endl;
  }
  writeSurface(domain);
  writeVolume(domain);

  // deposit dielectric
  {
    std::cout << "Dielectric Deposition ..." << std::flush;
    const NumericType thickness = 50;                   // nm
    domain->duplicateTopLevelSet(Material::Dielectric); // add layer
    auto dist = IsotropicProcessGeometric::New(thickness, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  // CMP at 90
  Planarize<NumericType, D>(domain, 90.0).apply();
  writeVolume(domain);

  // now remove gate and add new gate materials
  domain->removeMaterial(Material::PolySi);
  writeSurface(domain);
  writeVolume(domain);

  // now deposit TiN and PolySi as replacement gate
  {
    std::cout << "Gate Deposition TiN ..." << std::flush;
    const NumericType thickness = 4;             // nm
    domain->duplicateTopLevelSet(Material::TiN); // add layer
    auto dist = IsotropicProcessGeometric::New(thickness, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  {
    std::cout << "Gate Deposition PolySi ..." << std::flush;
    const NumericType thickness = 40;               // nm
    domain->duplicateTopLevelSet(Material::PolySi); // add layer
    auto dist = IsotropicProcessGeometric::New(thickness, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }

  // CMP at 90
  Planarize<NumericType, D>(domain, 90.0).apply();
  writeVolume(domain);

  domain->saveVolumeMesh("FinFET_Final", 0.05);
}
