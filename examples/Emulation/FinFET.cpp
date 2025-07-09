#include <geometries/psMakePlane.hpp>
#include <models/psAnisotropicProcess.hpp>
#include <models/psDirectionalProcess.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psIsotropicProcess.hpp>
#include <psDomain.hpp>
#include <psPlanarize.hpp>
#include <psProcess.hpp>
#include <psReader.hpp>
#include <psWriter.hpp>

#include <lsReader.hpp>

#include "epitaxy.hpp"

using namespace viennaps;

constexpr int D = 3;
using NumericType = double;
using LevelSetType = SmartPointer<viennals::Domain<NumericType, D>>;
using DomainType = SmartPointer<Domain<NumericType, D>>;
constexpr bool volumeOutput = true;

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
  NumericType bounds[2 * D] = {0, 90, 0, 100, 0, 70}; // in nanometers
  constexpr NumericType gridDelta = 0.59;
  auto domain = DomainType::New();

  // Initialise domain with a single silicon plane (at z=70 because it is 70
  // nm high)
  {
    auto ls = LevelSetType::New(bounds, boundaryConds, gridDelta);
    VectorType<double, D> origin{0, 0, 70};
    VectorType<double, D> normal{0, 0, 1};
    viennals::MakeGeometry<NumericType, D>(
        ls, SmartPointer<viennals::Plane<NumericType, D>>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(ls, Material::Si);
  }
  writeSurface(domain);

  // Add double patterning mask
  {
    auto ls = LevelSetType::New(domain->getGrid());
    VectorType<double, D> min{30, -10, 69.9};
    VectorType<double, D> max{60, 110, 90};
    viennals::MakeGeometry<NumericType, D> geometryFactory(
        ls, SmartPointer<viennals::Box<NumericType, D>>::New(min, max));
    geometryFactory.setIgnoreBoundaryConditions(
        true); // so that the mask is not mirrored inside domain at bounds
    geometryFactory.apply();

    domain->insertNextLevelSetAsMaterial(ls, Material::Mask);
  }
  writeSurface(domain);

  // Double patterning processes
  { // DP-Depo
    std::cout << "DP-Depo ...";
    domain->duplicateTopLevelSet(Material::Metal);
    auto dist =
        SmartPointer<SphereDistribution<NumericType, D>>::New(4, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  { // DP-Patterning
    std::cout << "DP-Patterning ...";
    auto dist = SmartPointer<BoxDistribution<NumericType, D>>::New(
        std::array<NumericType, D>{-gridDelta, -gridDelta, -6}, gridDelta,
        domain->getLevelSets().front());
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
    std::cout << "Si-Patterning ...";
    Vec3D<NumericType> direction = {0, 0, -1};
    auto model = SmartPointer<DirectionalProcess<NumericType, D>>::New(
        direction, -1.1, 0.1, Material::Metal, false);
    Process<NumericType, D>(domain, model, 90.).apply();
    std::cout << " done" << std::endl;
  }
  writeVolume(domain);
  writeSurface(domain);

  // Remove DP mask (metal)
  domain->removeTopLevelSet();
  writeSurface(domain);

  // deposit STI material
  {
    std::cout << "STI Deposition ...";
    domain->duplicateTopLevelSet(Material::SiO2);
    auto dist =
        SmartPointer<SphereDistribution<NumericType, D>>::New(35, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  // CMP at 80
  Planarize<NumericType, D>(domain, 80.0).apply();
  writeVolume(domain);

  // pattern STI material
  {
    std::cout << "STI Patterning ...";
    auto dist = SmartPointer<SphereDistribution<NumericType, D>>::New(
        -35, gridDelta, domain->getLevelSets().front());
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);
  writeVolume(domain);

  // deposit gate material
  {
    std::cout << "Gate Deposition HfO2 ...";
    domain->duplicateTopLevelSet(Material::HfO2);
    auto dist =
        SmartPointer<SphereDistribution<NumericType, D>>::New(2, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  {
    std::cout << "Gate Deposition PolySi ...";
    domain->duplicateTopLevelSet(Material::PolySi);
    auto dist =
        SmartPointer<SphereDistribution<NumericType, D>>::New(104, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  // CMP at 150
  Planarize<NumericType, D>(domain, 150.0).apply();

  // dummy gate mask addition
  {
    auto ls = LevelSetType::New(domain->getGrid());
    VectorType<double, D> min{-10, 30, 145};
    VectorType<double, D> max{100, 70, 175};
    viennals::MakeGeometry<NumericType, D> geometryFactory(
        ls, SmartPointer<viennals::Box<NumericType, D>>::New(min, max));
    geometryFactory.setIgnoreBoundaryConditions(
        true); // so that the mask is not mirrored inside domain at bounds
    geometryFactory.apply();

    domain->insertNextLevelSetAsMaterial(ls, Material::Mask);
  }
  writeSurface(domain);

  // gate patterning
  {
    std::cout << "Dummy Gate Patterning ...";
    Vec3D<NumericType> direction = {0, 0, -1};
    std::vector<Material> masks = {Material::Mask, Material::Si,
                                   Material::SiO2};
    auto model = SmartPointer<DirectionalProcess<NumericType, D>>::New(
        direction, -1.0, 0., masks, false);
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
    std::cout << "Spacer Deposition ...";
    domain->duplicateTopLevelSet(Material::Mask);
    auto dist =
        SmartPointer<SphereDistribution<NumericType, D>>::New(10, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  { // spacer etch
    std::cout << "Spacer Etch ...";
    auto ls = domain->getLevelSets()[domain->getLevelSets().size() - 2];
    auto dist = SmartPointer<BoxDistribution<NumericType, D>>::New(
        std::array<NumericType, D>{-gridDelta, -gridDelta, -50}, gridDelta, ls);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);
  writeVolume(domain);

  // isotropic etch (fin-release)
  {
    std::cout << "Fin-Release ...";
    std::vector<Material> masks = {Material::Mask, Material::SiGe,
                                   Material::PolySi, Material::HfO2,
                                   Material::SiO2};
    auto model =
        SmartPointer<IsotropicProcess<NumericType, D>>::New(-1., masks);
    Process<NumericType, D> process(domain, model, 5.);
    process.setIntegrationScheme(
        viennals::IntegrationSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER);
    process.apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);
  writeVolume(domain);

  // source/drain epitaxy
  {
    std::cout << "S/D Epitaxy ...";
    domain->duplicateTopLevelSet(Material::SiGe);

    auto maskLayer = LevelSetType::New(domain->getLevelSets().back());
    viennals::BooleanOperation(
        maskLayer, domain->getLevelSets().front(),
        viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
        .apply();

    auto tmpEpiDomain = DomainType::New();
    tmpEpiDomain->insertNextLevelSetAsMaterial(maskLayer, Material::Mask);
    tmpEpiDomain->insertNextLevelSetAsMaterial(domain->getLevelSets().back(),
                                               Material::SiGe, false);

    auto levelSets = tmpEpiDomain->getLevelSets();
    viennals::PrepareStencilLocalLaxFriedrichs(levelSets, {false, true});

    AdvectionParameters<NumericType> advectionParams;
    advectionParams.integrationScheme =
        viennals::IntegrationSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER;

    lsInternal::StencilLocalLaxFriedrichsScalar<NumericType, D,
                                                1>::setMaxDissipation(1000);

    std::vector<std::pair<Material, NumericType>> material = {
        {Material::SiGe, -1.}};
    auto model = SmartPointer<SelectiveEpitaxy<NumericType, D>>::New(material);
    Process<NumericType, D> proc(tmpEpiDomain, model, 15.);
    proc.setAdvectionParameters(advectionParams);
    proc.apply();

    viennals::FinalizeStencilLocalLaxFriedrichs(levelSets);
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);
  writeVolume(domain);

  // deposit dielectric
  {
    std::cout << "Dielectric Deposition ...";
    domain->duplicateTopLevelSet(Material::Dielectric);
    auto dist =
        SmartPointer<SphereDistribution<NumericType, D>>::New(50, gridDelta);
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
    std::cout << "Gate Deposition TiN ...";
    domain->duplicateTopLevelSet(Material::TiN);
    auto dist =
        SmartPointer<SphereDistribution<NumericType, D>>::New(4, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }
  writeSurface(domain);

  {
    std::cout << "Gate Deposition PolySi ...";
    domain->duplicateTopLevelSet(Material::PolySi);
    auto dist =
        SmartPointer<SphereDistribution<NumericType, D>>::New(40, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
    std::cout << " done" << std::endl;
  }

  // CMP at 90
  Planarize<NumericType, D>(domain, 90.0).apply();
  writeVolume(domain);

  domain->saveVolumeMesh("FinFET_Final", 0.05);

  return 0;
}