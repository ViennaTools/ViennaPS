#include <geometries/psMakePlane.hpp>
#include <models/psAnisotropicProcess.hpp>
#include <models/psDirectionalEtching.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psIsotropicProcess.hpp>
#include <psDomain.hpp>
#include <psPlanarize.hpp>
#include <psProcess.hpp>

using namespace viennaps;

constexpr int D = 3;
using NumericType = double;
using LevelSetType = SmartPointer<viennals::Domain<NumericType, D>>;
using DomainType = SmartPointer<Domain<NumericType, D>>;
constexpr NumericType gridDelta = 0.79;
unsigned outputNum = 0;
constexpr bool isEmulation = true;

// class epitaxy final : public lsVelocityField<NumericType> {
//   std::vector<double> velocities;
//   static constexpr double R111 = 0.5;
//   static constexpr double R100 = 1.;
//   static constexpr double low =
//       (D > 2) ? 0.5773502691896257 : 0.7071067811865476;
//   static constexpr double high = 1.0;

// public:
//   epitaxy(std::vector<double> vel) : velocities(vel){};

//   double getScalarVelocity(const std::array<NumericType, 3> & /*coordinate*/,
//                            int material,
//                            const std::array<NumericType, 3> &normal,
//                            unsigned long /* pointID */) final {
//     // double vel = std::max(std::max(std::abs(normal[0]),
//     std::abs(normal[1])),
//     // std::abs(normal[2]));
//     double vel = std::max(std::abs(normal[0]), std::abs(normal[2]));

//     constexpr double factor = (R100 - R111) / (high - low);
//     vel = (vel - low) * factor + R111;

//     if (std::abs(normal[0]) < std::abs(normal[2])) {
//       vel *= 2.;
//     }

//     return vel *
//            ((material < int(velocities.size())) ? velocities[material] : 0);
//   }
// };

void writeVolume(DomainType domain) {
  // return;// TODO: comment to write volumes
  static int volumeNum = 0;
  domain->saveVolumeMesh("FinFET_" + std::to_string(volumeNum));
  ++volumeNum;
}

void writeSurfaces(DomainType domain) {
  static int outputNum = 0;
  domain->saveSurfaceMesh("FinFET_" + std::to_string(outputNum) + ".vtp", true);
  ++outputNum;
}

int main() {

  NumericType bounds[2 * D] = {0, 90, 0, 100, 0, 70}; // in nanometres
  auto domain = DomainType::New();
  // Initialise domain with a single silicon plane (at z=70 because it is 70
  // nm high)
  MakePlane<NumericType, D>(domain, gridDelta, bounds, 70, true, Material::Si)
      .apply();
  writeSurfaces(domain);

  // Add double patterning mask
  {
    auto ls = LevelSetType::New(domain->getGrid());
    hrleVectorType<double, D> min(30, -10, 69.9);
    hrleVectorType<double, D> max(60, 110, 90);
    viennals::MakeGeometry<NumericType, D> geometryFactory(
        ls, SmartPointer<viennals::Box<NumericType, D>>::New(min, max));
    geometryFactory.setIgnoreBoundaryConditions(
        true); // so that the mask is not mirrored inside domain at bounds
    geometryFactory.apply();

    domain->insertNextLevelSetAsMaterial(ls, Material::Mask);
  }
  writeSurfaces(domain);

  // Double patterning processes
  { // DP-Depo
    domain->duplicateTopLevelSet(Material::SiO2);
    auto dist =
        SmartPointer<SphereDistribution<NumericType, D>>::New(4, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
  }
  writeSurfaces(domain);

  { // DP-Patterning
    auto dist = SmartPointer<BoxDistribution<NumericType, D>>::New(
        std::array<NumericType, D>{-gridDelta, -gridDelta, -6}, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
  }
  writeSurfaces(domain);

  // Remove mask with boolean op
  domain->removeMaterial(Material::Mask);
  writeSurfaces(domain);
  writeVolume(domain);

  // pattern si/sige/si stack
  {
    Vec3D<NumericType> direction = {0, 0, -1};
    auto model = SmartPointer<DirectionalEtching<NumericType, D>>::New(
        direction, -1.0, 0.1, Material::SiO2, false);
    Process<NumericType, D>(domain, model, 90.).apply();
  }
  writeVolume(domain);

  // Remove DP mask
  domain->removeTopLevelSet();
  writeSurfaces(domain);

  // deposit STI material
  {
    domain->duplicateTopLevelSet(Material::SiO2);
    auto dist =
        SmartPointer<SphereDistribution<NumericType, D>>::New(35, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
  }
  writeSurfaces(domain);

  // CMP at 80
  Planarize<NumericType, D>(domain, 80.0).apply();
  writeVolume(domain);

  // pattern STI material
  {
    auto dist = SmartPointer<SphereDistribution<NumericType, D>>::New(
        -35, gridDelta, domain->getLevelSets().front());
    Process<NumericType, D>(domain, dist).apply();
  }
  writeSurfaces(domain);
  writeVolume(domain);

  // deposit gate material
  {
    domain->duplicateTopLevelSet(Material::HfO2);
    auto dist =
        SmartPointer<SphereDistribution<NumericType, D>>::New(2, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
  }
  writeSurfaces(domain);

  {
    domain->duplicateTopLevelSet(Material::PolySi);
    auto dist =
        SmartPointer<SphereDistribution<NumericType, D>>::New(104, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
  }
  writeSurfaces(domain);

  // CMP at 150
  Planarize<NumericType, D>(domain, 150.0).apply();

  // dummy gate mask addition
  {
    auto ls = LevelSetType::New(domain->getGrid());
    hrleVectorType<double, D> min(-10, 30, 145);
    hrleVectorType<double, D> max(100, 70, 175);
    viennals::MakeGeometry<NumericType, D> geometryFactory(
        ls, SmartPointer<viennals::Box<NumericType, D>>::New(min, max));
    geometryFactory.setIgnoreBoundaryConditions(
        true); // so that the mask is not mirrored inside domain at bounds
    geometryFactory.apply();

    domain->insertNextLevelSetAsMaterial(ls, Material::Mask);
  }
  writeSurfaces(domain);

  // gate patterning
  {
    Vec3D<NumericType> direction = {0, 0, -1};
    std::vector<Material> masks = {Material::Mask, Material::Si};
    auto model = SmartPointer<DirectionalEtching<NumericType, D>>::New(
        direction, -1.0, 0., masks, false);
    Process<NumericType, D>(domain, model, 110.).apply();
  }

  // Remove mask
  domain->removeTopLevelSet();
  writeSurfaces(domain);
  writeVolume(domain);

  // Spacer Deposition and Etch
  { // spacer depo
    domain->duplicateTopLevelSet(Material::Mask);
    auto dist =
        SmartPointer<SphereDistribution<NumericType, D>>::New(10, gridDelta);
    Process<NumericType, D>(domain, dist).apply();
  }
  writeSurfaces(domain);

  { // spacer etch
    auto ls = domain->getLevelSets()[domain->getLevelSets().size() - 2];
    auto dist = SmartPointer<BoxDistribution<NumericType, D>>::New(
        std::array<NumericType, D>{-gridDelta, -gridDelta, -50}, gridDelta, ls);
    Process<NumericType, D>(domain, dist).apply();
  }
  writeVolume(domain);

  // isotropic etch (fin-release)
  {
    std::vector<Material> masks = {Material::Mask, Material::SiGe,
                                   Material::PolySi, Material::HfO2,
                                   Material::SiO2};
    auto model =
        SmartPointer<IsotropicProcess<NumericType, D>>::New(-1., masks);
    Process<NumericType, D>(domain, model, 5.).apply();
  }
  writeSurfaces(domain);
  writeVolume(domain);

  // source/drain epitaxy
  {
    domain->duplicateTopLevelSet(Material::SiGe);
    std::vector<std::pair<Material, NumericType>> material = {
        {Material::SiGe, 1.}, {Material::Si, 1.}};
    auto model =
        SmartPointer<AnisotropicProcess<NumericType, D>>::New(material);
    Process<NumericType, D> proc(domain, model, 15.);
    proc.setIntegrationScheme(viennals::IntegrationSchemeEnum::
                                  STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER);
    proc.apply();
  }
  writeSurfaces(domain);
  writeVolume(domain);

  // // {
  // //   processes.clear();
  // //   auto epiVel = SmartPointer<epitaxy>::New(
  // //       std::vector<double>({1, 0, 0, 0, 0, 0, 1}));
  // //   auto isoVel = SmartPointer<isotropic>::New(
  // //       std::vector<double>({1, 0, 0, 0, 0, 0, 1}));
  // //   processes.push_back(Process("S/D-Epitaxy", 15, epiVel, true));
  // // }
  // // execute(domains, processes);
  // // writeVolume(domains);

  // // deposit dielectric
  // if constexpr (isEmulation) {
  //   domains.push_back(LevelSetType::New(domains.back()));
  //   auto dist =
  //       SmartPointer<lsSphereDistribution<NumericType, D>>::New(50,
  //       gridDelta);
  //   emulate("Dielectric-Depo", domains.back(), dist);
  //   writeSurfaces(domains);
  // } else {
  //   processes.clear();
  //   processes.push_back(Process("Dielectric-Deposit", 50, isoVelocity,
  //   true)); execute(domains, processes);
  // }

  // // CMP at 90
  // planarise(domains, 90.0);

  // writeVolume(domains);

  // // now remove gate and add new gate materials
  // {
  //   // bool dummy gate from all higher layers
  //   // but keep union with layer below for correct wrapping
  //   unsigned dummyGateID = 3;
  //   for (unsigned i = dummyGateID + 1; i < domains.size(); ++i) {
  //     lsBooleanOperation(domains[i], domains[dummyGateID],
  //                        lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
  //         .apply();

  //     lsBooleanOperation(domains[i], domains[dummyGateID - 1],
  //                        lsBooleanOperationEnum::UNION)
  //         .apply();
  //   }
  //   domains.erase(domains.begin() + dummyGateID);
  //   writeSurfaces(domains);
  //   writeVolume(domains);
  // }

  // // now deposit TiN and PolySi as replacement gate
  // if constexpr (isEmulation) {
  //   {
  //     domains.push_back(LevelSetType::New(domains.back()));
  //     auto dist =
  //         SmartPointer<lsSphereDistribution<NumericType, D>>::New(4,
  //         gridDelta);
  //     emulate("TiN-Deposit", domains.back(), dist);
  //     writeSurfaces(domains);
  //   }
  //   {
  //     domains.push_back(LevelSetType::New(domains.back()));
  //     auto dist = SmartPointer<lsSphereDistribution<NumericType, D>>::New(
  //         40, gridDelta);
  //     emulate("PolySi-Deposit", domains.back(), dist);
  //     writeSurfaces(domains);
  //   }
  // } else {
  //   processes.clear();
  //   processes.push_back(Process("TiN-Deposit", 4, isoVelocity, true));
  //   processes.push_back(Process("PolySi-Deposit", 40, isoVelocity, true));
  //   execute(domains, processes);
  // }

  // // CMP at 90
  // planarise(domains, 89.0);

  // writeVolume(domains);

  return 0;
}