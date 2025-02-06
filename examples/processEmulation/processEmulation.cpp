#include <array>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>

#include <lsAdvect.hpp>
#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsExpand.hpp>
#include <lsGeometricAdvect.hpp>
#include <lsGeometries.hpp>
#include <lsMakeGeometry.hpp>
#include <lsPrune.hpp>
#include <lsSmartPointer.hpp>
#include <lsToMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>
#include <lsWriteVisualizationMesh.hpp>
#include <lsWriter.hpp>

#include <geometries/psMakePlane.hpp>
#include <psDomain.hpp>
#include <psProcess.hpp>

using namespace viennaps;

constexpr int D = 3;
using NumericType = double;
constexpr NumericType gridDelta = 1.07; // 0.51;
unsigned outputNum = 0;
// using LevelSetType = lsSmartPointer<lsDomain<NumericType, D>>;

// bool exists(const std::string &name) {
//   std::ifstream f(name.c_str());
//   return f.good();
// }

// // Implement own velocity field
// class isotropic final : public lsVelocityField<NumericType> {
//   std::vector<double> velocities;

// public:
//   isotropic(std::vector<double> vel) : velocities(vel){};

//   double getScalarVelocity(const std::array<NumericType, 3> & /*coordinate*/,
//                            int material,
//                            const std::array<NumericType, 3> &
//                            /*normalVector*/, unsigned long pointID) final {
//     return (material < int(velocities.size())) ? velocities[material] : 0;
//   }
// };

// // Directional etch for one material
// class directional final : public lsVelocityField<NumericType> {
//   const std::array<NumericType, D> direction;
//   std::vector<double> velocities;
//   const double isoVelocity;

// public:
//   directional(std::array<NumericType, D> dir, std::vector<double> vel,
//               double isoVel = 0)
//       : direction(dir), velocities(vel), isoVelocity(isoVel){};

//   std::array<double, D>
//   getVectorVelocity(const std::array<NumericType, 3> &coordinate, int
//   material,
//                     const std::array<NumericType, 3> &normalVector,
//                     unsigned long pointID) final {
//     if (material < int(velocities.size())) {
//       std::array<NumericType, D> dir(direction);
//       for (unsigned i = 0; i < D; ++i) {
//         dir[i] *= velocities[material];
//         dir[i] += isoVelocity;
//       }
//       // if(coordinate == std::array<NumericType, 3>({10, 0, 28})){
//       //   std::cout << "OUTPUT " << outputNum << " -- > Material: " <<
//       material
//       //   << std::endl; std::cout << "Normal: " << normalVector[0] << ", "
//       <<
//       //   normalVector[1] << ", " << normalVector[2] << std::endl; std::cout
//       <<
//       //   "Velocity: " << dir[0] << ", " << dir[1] << ", " << dir[2] <<
//       //   std::endl;
//       // }
//       return dir;
//     } else {
//       return {0};
//     }
//   }
// };

// // double maxVel = 0.;

// // Implement own velocity field
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
//     double vel = std::max(std::max(std::abs(normal[0]), std::abs(normal[1])),
//                           std::abs(normal[2]));
//     // double vel = std::max(std::abs(normal[0]), std::abs(normal[2]));

//     constexpr double factor = (R100 - R111) / (high - low);
//     vel = (vel - low) * factor + R111;

//     if (std::abs(normal[0]) < std::abs(normal[2])) {
//       vel *= 2.;
//     }

//     // maxVel = std::max(maxVel, -vel);
//     return vel *
//            ((material < int(velocities.size())) ? velocities[material] : 0);
//   }
// };

// void writeSurfaces(std::deque<LevelSetType> &domains,
//                    bool writePoints = false) {
//   static unsigned numMat = 0;

//   for (unsigned i = 0; (i < domains.size()) || (i < numMat); ++i) {
//     auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
//     auto pointMesh = lsSmartPointer<lsMesh<NumericType>>::New();
//     if (numMat != 0 && i >= numMat) { // write emtpy meshes before so
//     paraview
//                                       // displays it correctly
//       for (unsigned j = 0; j < outputNum; ++j) {
//         std::string fileName =
//             std::to_string(i) + "-" + std::to_string(j) + ".vtk";
//         if (!exists(fileName)) {
//           lsVTKWriter<NumericType>(mesh, "surface-m" + fileName).apply();
//           lsVTKWriter<NumericType>(mesh, "points-m" + fileName).apply();
//         }
//       }
//     }

//     if (i < domains.size()) {
//       lsToSurfaceMesh<NumericType, D>(domains[i], mesh).apply();
//       if (writePoints) {
//         lsToMesh<NumericType, D>(domains[i], pointMesh).apply();
//       }
//     }
//     std::string fileName = "surface-m" + std::to_string(i) + "-" +
//                            std::to_string(outputNum) + ".vtk";
//     lsVTKWriter<NumericType>(mesh, fileName).apply();
//     // fileName = "levelset-m" + std::to_string(i) + "-" +
//     // std::to_string(outputNum) + ".lvst"; lsWriter<double, D>(domains[i],
//     // fileName).apply();
//     if (writePoints) {
//       std::string pointName = "points-m" + std::to_string(i) + "-" +
//                               std::to_string(outputNum) + ".vtk";
//       lsVTKWriter<NumericType>(pointMesh, pointName).apply();
//     }
//   }
//   numMat = (domains.size() < numMat) ? numMat : domains.size();
//   // increase count
//   ++outputNum;
// }

// void writeVolume(std::deque<LevelSetType> &domains) {
//   // return;// TODO: comment to write volumes
//   static int volumeNum = 0;
//   auto volumeMeshing =
//       lsSmartPointer<lsWriteVisualizationMesh<NumericType, D>>::New();
//   for (auto &it : domains) {
//     volumeMeshing->insertNextLevelSet(it);
//   }
//   volumeMeshing->setFileName("MBCFET" + std::to_string(volumeNum));
//   volumeMeshing->apply();
//   ++volumeNum;
// }

// class Process {
// public:
//   std::string name;
//   double time;
//   lsSmartPointer<lsVelocityField<NumericType>> velocities;
//   bool newLayer;

//   template <class VelocityField>
//   Process(std::string processName, double processTime,
//           VelocityField processVelocities, bool newMaterial = false)
//       : name(processName), time(processTime), newLayer(newMaterial) {
//     velocities = std::dynamic_pointer_cast<lsVelocityField<NumericType>>(
//         processVelocities);
//   }
// };

// void execute(std::deque<LevelSetType> &domains,
//              std::vector<Process> &processes) {
//   // Process loop
//   for (auto &it : processes) {
//     std::cout << "Running " << it.name << " for " << it.time << "s"
//               << std::endl;
//     if (it.newLayer) { // if new layer, copy last level set
//       std::cout << "Adding new layer" << std::endl;
//       domains.push_back(LevelSetType::New(domains.back()));
//     }

//     auto advectionKernel = lsSmartPointer<lsAdvect<NumericType, D>>::New();
//     advectionKernel->setVelocityField(it.velocities);
//     for (auto &it : domains) {
//       advectionKernel->insertNextLevelSet(it);
//     }

//     advectionKernel->setAdvectionTime(it.time);
//     advectionKernel->apply();

//     writeSurfaces(domains, true);
//   }
// }

// void planarise(std::deque<LevelSetType> &domains, double height,
//                bool addLayer = false) {
//   auto plane = LevelSetType::New(domains.back()->getGrid()); // empty domain
//   double origin[3] = {0., (D == 2) ? height : 0., (D == 3) ? height : 0.};
//   double planeNormal[3] = {0., (D == 2) ? 1 : 0., (D == 3) ? 1 : 0.};
//   lsMakeGeometry<NumericType, D>(
//       plane, lsSmartPointer<lsPlane<NumericType, D>>::New(origin,
//       planeNormal)) .apply();
//   // now remove plane from domain
//   for (auto &it : domains) {
//     lsBooleanOperation<NumericType, D>(it, plane,
//                                        lsBooleanOperationEnum::INTERSECT)
//         .apply();
//   }

//   if (addLayer) {
//     domains.push_back(plane);
//   }

//   writeSurfaces(domains, true);
// }

// template <class DIST>
// void emulate(LevelSetType surface, DIST distribution, std::string
// processName,
//              LevelSetType mask = nullptr) {
//   auto start = std::chrono::high_resolution_clock::now();
//   if (mask == nullptr) {
//     lsGeometricAdvect<NumericType, D>(surface, distribution).apply();
//   } else {
//     lsGeometricAdvect<NumericType, D>(surface, distribution, mask).apply();
//   }
//   auto end = std::chrono::high_resolution_clock::now();
//   auto duration =
//       std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//   std::cout << processName << " took " << (duration.count() / 1000.) << " s"
//             << std::endl;
// }

int main() {
  omp_set_num_threads(16);

  NumericType bounds[2 * D] = {0, 70, 0, 100, 0, 70}; // in nanometres

  auto domain = SmartPointer<Domain<NumericType, D>>::New();
  MakePlane<NumericType, D>(domain, gridDelta, 70., 100., 10., true,
                            Material::Si)
      .apply();
  // TODO: adjust bounds

  //   writeSurfaces(domains, true);

  //   std::vector<Process> processes;
  //   auto isoVelocity =
  //       lsSmartPointer<isotropic>::New(std::vector<double>(10, 1.0));

  //   // Epi Fin growth
  //   {
  //     // SiGe
  //     {
  //       // copy last layer
  //       domains.push_back(LevelSetType::New(domains.back()));
  //       auto dist = lsSmartPointer<lsSphereDistribution<NumericType,
  //       D>>::New(
  //           7, gridDelta);
  //       emulate(domains.back(), dist, "SiGe-Epitaxy");
  //     }
  //     // Si
  //     {
  //       // copy last layer
  //       domains.push_back(LevelSetType::New(domains.back()));
  //       auto dist = lsSmartPointer<lsSphereDistribution<NumericType,
  //       D>>::New(
  //           5, gridDelta);
  //       emulate(domains.back(), dist, "Si-Epitaxy");
  //     }
  //     // SiGe
  //     {
  //       // copy last layer
  //       domains.push_back(LevelSetType::New(domains.back()));
  //       auto dist = lsSmartPointer<lsSphereDistribution<NumericType,
  //       D>>::New(
  //           7, gridDelta);
  //       emulate(domains.back(), dist, "SiGe-Epitaxy");
  //     }
  //     // Si
  //     {
  //       // copy last layer
  //       domains.push_back(LevelSetType::New(domains.back()));
  //       auto dist = lsSmartPointer<lsSphereDistribution<NumericType,
  //       D>>::New(
  //           5, gridDelta);
  //       emulate(domains.back(), dist, "Si-Epitaxy");
  //     }
  //     // SiGe
  //     {
  //       // copy last layer
  //       domains.push_back(LevelSetType::New(domains.back()));
  //       auto dist = lsSmartPointer<lsSphereDistribution<NumericType,
  //       D>>::New(
  //           7, gridDelta);
  //       emulate(domains.back(), dist, "SiGe-Epitaxy");
  //     }
  //     // Si
  //     {
  //       // copy last layer
  //       domains.push_back(LevelSetType::New(domains.back()));
  //       auto dist = lsSmartPointer<lsSphereDistribution<NumericType,
  //       D>>::New(
  //           5, gridDelta);
  //       emulate(domains.back(), dist, "Si-Epitaxy");
  //     }
  //   }
  //   writeSurfaces(domains, true);

  //   // Add nanosheet mask
  //   {
  //     domains.push_back(LevelSetType::New(domains.back()->getGrid()));
  //     hrleVectorType<double, D> min(10, bounds[2] - 10, 34.0);
  //     hrleVectorType<double, D> max(60, bounds[3] + 10, 70.0);
  //     lsMakeGeometry<NumericType, D> geometryFactory(
  //         domains.back(), lsSmartPointer<lsBox<NumericType, D>>::New(min,
  //         max));
  //     bool ignoreBNC = true;
  //     geometryFactory.setIgnoreBoundaryConditions(
  //         ignoreBNC); // so that the mask is not mirrored inside domain
  //         at bounds
  //     geometryFactory.apply();
  //     // wrap around other LSs
  //     lsBooleanOperation(domains.back(), *(domains.end() - 2),
  //                        lsBooleanOperationEnum::UNION)
  //         .apply();
  //   }
  //   writeSurfaces(domains, true);

  //   // pattern si/sige/si stack
  //   // pattern si Fins
  //   {
  //     processes.clear();
  //     std::array<NumericType, D> direction = {0, 0, -1};
  //     auto directVelocity = lsSmartPointer<directional>::New(
  //         direction, std::vector<double>(7, 1), 0.01);
  //     processes.push_back(Process("Si/SiGe-Patterning", 80,
  //     directVelocity));
  //   }
  //   execute(domains, processes);

  //   // Remove DP mask
  //   domains.pop_back();
  //   writeSurfaces(domains, true);

  //   // deposit and cmp STI
  //   {
  //     // push back new domain
  //     domains.push_back(lsSmartPointer<lsDomain<NumericType, D>>::New(
  //         domains.back()->getGrid()));

  //     NumericType origin[D] = {0., 0., 60.};
  //     NumericType orientation[D] = {0., 0., 1.};
  //     lsMakeGeometry<NumericType, D>(
  //         domains.back(),
  //         lsSmartPointer<lsPlane<NumericType, D>>::New(origin,
  //         orientation)) .apply();
  //     writeSurfaces(domains);
  //   }
  //   writeVolume(domains);

  //   // Pattern STI
  //   {
  //     // selective isotropic etch for 60 nm
  //     auto dist = lsSmartPointer<lsSphereDistribution<NumericType,
  //     D>>::New(
  //         -50, gridDelta);

  //     emulate(domains.back(), dist, "STI-Pattern", *(domains.end() - 2));

  //     writeSurfaces(domains);
  //   }
  //   writeVolume(domains);

  //   // deposit and CMP Dummy gate
  //   {
  //     // push back new domain
  //     domains.push_back(lsSmartPointer<lsDomain<NumericType, D>>::New(
  //         domains.back()->getGrid()));

  //     NumericType origin[D] = {0., 0., 150.};
  //     NumericType orientation[D] = {0., 0., 1.};
  //     lsMakeGeometry<NumericType, D>(
  //         domains.back(),
  //         lsSmartPointer<lsPlane<NumericType, D>>::New(origin,
  //         orientation)) .apply();
  //     writeSurfaces(domains);
  //   }

  //   // dummy gate mask addition
  //   {
  //     domains.push_back(LevelSetType::New(domains.back()->getGrid()));
  //     hrleVectorType<double, D> min(bounds[0] - 10, bounds[2] + 30, 145);
  //     hrleVectorType<double, D> max(bounds[1] + 10, bounds[3] - 30, 165);
  //     lsMakeGeometry<NumericType, D> geometryFactory(
  //         domains.back(), lsSmartPointer<lsBox<NumericType, D>>::New(min,
  //         max));
  //     bool ignoreBNC = true;
  //     geometryFactory.setIgnoreBoundaryConditions(
  //         ignoreBNC); // so that the mask is not mirrored inside domain
  //         at bounds
  //     geometryFactory.apply();

  //     lsBooleanOperation<NumericType, D>(*(domains.end() - 2),
  //     domains.back(),
  //                                        lsBooleanOperationEnum::UNION)
  //         .apply();
  //     // // wrap all other layers accordingly
  //     // for (unsigned i = 1; i < domains.size(); ++i) { // only bool
  //     with 5th
  //     // layer
  //     //   lsBooleanOperation<NumericType, D>(domains[i],
  //     domains.front(),
  //     // lsBooleanOperationEnum::UNION)
  //     //       .apply();
  //     // }
  //   }
  //   writeSurfaces(domains, true);

  //   // dummy gate patterning
  //   {
  //     std::array<NumericType, 3> etchRates{-gridDelta, -gridDelta, -145};
  //     auto dist = lsSmartPointer<lsBoxDistribution<NumericType, D>>::New(
  //         etchRates, gridDelta);

  //     // make mask
  //     auto mask = LevelSetType::New(domains.back());
  //     lsBooleanOperation(mask, domains[7],
  //     lsBooleanOperationEnum::UNION).apply(); emulate(*(domains.end() -
  //     2), dist, "Gate Patterning", mask);

  //     // Remove mask
  //     {
  //       domains.pop_back(); // now remove
  //     }

  //     // writeSurfaces(domains);

  //     // for(unsigned i = 0; i < domains.size()-1; ++i) {
  //     //   lsBooleanOperation(domains[i], domains.back(),
  //     //   lsBooleanOperationEnum::INTERSECT).apply();
  //     // }
  //   }
  //   writeSurfaces(domains);
  //   writeVolume(domains);

  //   // Spacer Deposition
  //   {
  //     domains.push_back(LevelSetType::New(domains.back()));
  //     auto dist = lsSmartPointer<lsSphereDistribution<NumericType,
  //     D>>::New(
  //         10., gridDelta);
  //     emulate(domains.back(), dist, "Spacer-Depo");
  //   }
  //   writeSurfaces(domains);

  //   // Spacer Etch
  //   {
  //     // make mask
  //     auto mask = LevelSetType::New(*(domains.end() - 2));
  //     lsBooleanOperation(mask, *(domains.end() - 4),
  //                        lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
  //         .apply();
  //     lsBooleanOperation(mask, domains[0],
  //     lsBooleanOperationEnum::UNION).apply();

  //     {
  //       auto mesh = lsSmartPointer<lsMesh<>>::New();
  //       lsToSurfaceMesh(mask, mesh).apply();
  //       lsVTKWriter(mesh, "mask.vtp").apply();
  //     }

  //     auto dist = lsSmartPointer<lsBoxDistribution<NumericType, D>>::New(
  //         std::array<NumericType, D>{-gridDelta, -gridDelta, -50},
  //         gridDelta);
  //     emulate(domains.back(), dist, "Spacer-Patterning", mask);
  //     writeSurfaces(domains);

  //     for (unsigned i = 0; i < domains.size() - 1; ++i) {
  //       lsBooleanOperation(domains[i], domains.back(),
  //                          lsBooleanOperationEnum::INTERSECT)
  //           .apply();
  //     }
  //     writeSurfaces(domains);
  //   }
  //   writeVolume(domains);

  //   // Inner Spacers!!!
  //   // {
  //   //   processes.clear();
  //   //   // just etch SiGe for a little bit, so inner spacers are
  //   deposited when
  //   //   spacer is created auto SiGeEtch =
  //   lsSmartPointer<isotropic>::New(
  //   //       std::vector<double>({0, -1, 0, -1, 0, -1, 0}));

  //   //   processes.push_back(Process("InnerSpacer-Etch", 10, SiGeEtch));
  //   //   execute(domains, processes);
  //   // }
  //   {
  //     auto mask = LevelSetType::New(domains.back());
  //     // remove layers to etch from this material
  //     for (int i = 5; i >= 1; i -= 2) {
  //       auto temp = LevelSetType::New(domains[i]);
  //       lsBooleanOperation(temp, domains[i - 1],
  //                          lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
  //           .apply();
  //       lsBooleanOperation(mask, temp,
  //                          lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
  //           .apply();
  //     }
  //     // {
  //     //   auto mesh = lsSmartPointer<lsMesh<>>::New();
  //     //   lsToSurfaceMesh(mask, mesh).apply();
  //     //   lsVTKWriter(mesh, "mask.vtp").apply();
  //     // }
  //     auto dist = lsSmartPointer<lsSphereDistribution<NumericType,
  //     D>>::New(
  //         -10, gridDelta);

  //     emulate(domains.back(), dist, "InnerSpacerCavity-Etch", mask);

  //     for (unsigned i = 0; i < domains.size() - 1; ++i) {
  //       lsBooleanOperation(domains[i], domains.back(),
  //                          lsBooleanOperationEnum::INTERSECT)
  //           .apply();
  //     }

  //     writeSurfaces(domains);
  //   }
  //   writeVolume(domains);

  //   // deposit inner spacer
  //   {
  //     domains.push_back(LevelSetType::New(domains.back()));
  //     auto dist =
  //         lsSmartPointer<lsSphereDistribution<NumericType, D>>::New(5,
  //         gridDelta);
  //     emulate(domains.back(), dist, "InnerSpacer-Depo");
  //   }
  //   writeSurfaces(domains);

  //   // etch inner spacer
  //   {
  //     auto dist = lsSmartPointer<lsBoxDistribution<NumericType, D>>::New(
  //         std::array<NumericType, D>{-5, -5, -5}, gridDelta);
  //     emulate(domains.back(), dist, "InnerSpacer-Etch");
  //   }
  //   writeSurfaces(domains);
  //   writeVolume(domains);

  //   // source/drain epitaxy
  //   {
  //     processes.clear(); // 0  1  2  3  4  5  6

  //     // generate the top layer and mask layer used for layer wrapping
  //     auto maskLayer = LevelSetType::New(domains.back());
  //     lsBooleanOperation(maskLayer, domains[1],
  //                        lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
  //         .apply();

  //     // {
  //     //   auto mesh = lsSmartPointer<lsMesh<>>::New();
  //     //   lsToSurfaceMesh(maskLayer, mesh).apply();
  //     //   lsVTKWriter(mesh, "mask.vtp").apply();
  //     // }

  //     auto topLayer = LevelSetType::New(domains.back());
  //     std::vector<LevelSetType> LSs;
  //     LSs.push_back(maskLayer);
  //     LSs.push_back(topLayer);
  //     std::vector<bool> depoLayers = {false, true};
  //     lsPrepareStencilLocalLaxFriedrichs(LSs, depoLayers);

  //     std::cout << "S/D Epitaxy";

  //     auto epiVel = lsSmartPointer<epitaxy>::New(std::vector<double>({0,
  //     -1}));

  //     lsAdvect<NumericType, D> kernel(LSs, epiVel);
  //     // kernel.setIntegrationScheme(
  //     //
  //     lsIntegrationSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER);
  //     kernel.setSingleStep(true);

  //     auto start = std::chrono::high_resolution_clock::now();
  //     for (double time = 27.; time >= 0.1; time -=
  //     kernel.getAdvectedTime())
  //     {
  //       kernel.setAdvectionTime(time);
  //       kernel.apply();
  //       // std::cout << time << ": Epitaxy for " <<
  //       kernel.getAdvectedTime()
  //       << "
  //       // with " << maxVel << std::endl; maxVel = 0.;
  //     }
  //     auto end = std::chrono::high_resolution_clock::now();
  //     auto duration =
  //         std::chrono::duration_cast<std::chrono::milliseconds>(end -
  //         start);
  //     std::cout << " took " << (duration.count() / 1000.) << " s" <<
  //     std::endl;

  //     lsFinalizeStencilLocalLaxFriedrichs(LSs);
  //     domains.push_back(std::move(LSs.back()));
  //     lsExpand(domains[8], 3).apply();

  //     writeSurfaces(domains);
  //   }

  //   // Deposit and CMP Dielectric
  //   planarise(domains, 75., true);
  //   writeVolume(domains);

  //   // now remove gate
  //   {
  //     // bool dummy gate from all higher layers
  //     // but keep union with layer below for correct wrapping
  //     unsigned dummyGateID = 8;
  //     for (unsigned i = dummyGateID + 1; i < domains.size(); ++i) {
  //       lsBooleanOperation(domains[i], domains[dummyGateID],
  //                          lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
  //           .apply();

  //       lsBooleanOperation(domains[i], domains[dummyGateID - 1],
  //                          lsBooleanOperationEnum::UNION)
  //           .apply();
  //     }
  //     // deep copy empty domain, so layer numbering stays the same
  //     domains[dummyGateID]->deepCopy(LevelSetType::New(domains[dummyGateID
  //     - 1]));
  //     // .erase(domains.begin() + dummyGateID);
  //     writeSurfaces(domains);
  //     // writeVolume(domains);
  //   }
  //   writeVolume(domains);

  //   // Fin release
  //   {
  //     auto mask = LevelSetType::New(domains.back());
  //     // remove layers to etch from this material
  //     for (int i = 5; i >= 1; i -= 2) {
  //       auto temp = LevelSetType::New(domains[i]);
  //       lsBooleanOperation(temp, domains[i - 1],
  //                          lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
  //           .apply();
  //       lsBooleanOperation(mask, temp,
  //                          lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
  //           .apply();
  //     }

  //     {
  //       auto mesh = lsSmartPointer<lsMesh<>>::New();
  //       lsToSurfaceMesh(mask, mesh).apply();
  //       lsVTKWriter(mesh, "mask.vtp").apply();
  //     }

  //     auto dist = lsSmartPointer<lsSphereDistribution<NumericType,
  //     D>>::New(
  //         -30, gridDelta);

  //     emulate(domains.back(), dist, "Sheet-Release", mask);

  //     for (unsigned i = 0; i < domains.size() - 1; ++i) {
  //       lsBooleanOperation(domains[i], domains.back(),
  //                          lsBooleanOperationEnum::INTERSECT)
  //           .apply();
  //     }

  //     writeSurfaces(domains);
  //   }
  //   writeVolume(domains);

  //   // RMG sequence
  //   {
  //     // HfO2
  //     {
  //       domains.push_back(LevelSetType::New(domains.back()));
  //       auto dist = lsSmartPointer<lsSphereDistribution<NumericType,
  //       D>>::New(
  //           2, gridDelta);
  //       emulate(domains.back(), dist, "HfO2-Deposit");
  //       writeSurfaces(domains);
  //     }
  //     // TiN
  //     {
  //       domains.push_back(LevelSetType::New(domains.back()));
  //       auto dist = lsSmartPointer<lsSphereDistribution<NumericType,
  //       D>>::New(
  //           6, gridDelta);
  //       emulate(domains.back(), dist, "TiN-Deposit");
  //       writeSurfaces(domains);
  //     }
  //     // PolySi
  //     {
  //       domains.push_back(LevelSetType::New(domains.back()));
  //       auto dist = lsSmartPointer<lsSphereDistribution<NumericType,
  //       D>>::New(
  //           40, gridDelta);
  //       emulate(domains.back(), dist, "PolySi-Deposit");
  //       writeSurfaces(domains);
  //     }
  //   }

  //   // CMP metal gate
  //   planarise(domains, 74.);
  //   writeVolume(domains);

  return 0;
}