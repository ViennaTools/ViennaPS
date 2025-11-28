/**
  This example requires the FDTD library from https://github.com/exilief/FiDiTi
  to be installed and the install location passed to the build system
  (VIENNAPS_LOOKUP_DIRS)
*/

#include "FDTD.hpp"
#include "geometry.hpp"

#include <models/psSF6O2Etching.hpp>

#include <process/psProcess.hpp>
#include <vcUtil.hpp>

namespace ps = viennaps;
namespace cs = viennacs;

int main(int argc, char *argv[]) {
  using Scalar = double;
  constexpr int D = 2;

  ps::Logger::setLogLevel(ps::LogLevel::INTERMEDIATE);

  // Parse the parameters
  ps::util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    // Try default config file
    params.readConfigFile("config.txt");
    if (params.m.empty()) {
      std::cout << "No configuration file provided!" << std::endl;
      std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
      return 1;
    }
  }

  omp_set_num_threads(params.get<int>("numThreads"));

  bool withEtching = params.get("withEtching");

  // Set parameter units
  ps::units::Length::setUnit("um");
  ps::units::Time::setUnit("s");

  // Geometry setup
  auto domain = makeGeometry<Scalar, D>(
      params.get("xExtent"), params.get("yExtent"), params.get("gridDelta"),
      params.get("bulkHeight"), params.get("numHoles"), params.get("holeWidth"),
      params.get("holeDepth"), params.get("passivationHeight"),
      params.get("antiReflectHeight1"), params.get("antiReflectHeight2"),
      params.get("maskHeight"), 0. /*baseHeight*/, withEtching);

  if (withEtching) {
    // Save passivation & anti-reflect layers to re-add after etching
    auto [extraLayers, extraMaterials] = extractTopLevelSets(*domain, 3);

    // SF6O2 etching model
    auto model = ps::SmartPointer<ps::SF6O2Etching<Scalar, D>>::New(
        params.get("ionFlux"), params.get("etchantFlux"),
        params.get("oxygenFlux"), params.get("meanEnergy"),
        params.get("sigmaEnergy"),
        params.get("ionExponent") /*source power cosine distribution exponent*/,
        params.get("A_O") /*oxy sputter yield*/,
        params.get("etchStopDepth") /*max etch depth*/);

    // Process setup
    ps::Process<Scalar, D> process;
    process.setDomain(domain);
    process.setProcessModel(model);
    process.setMaxCoverageInitIterations(10);
    process.setNumberOfRaysPerPoint(params.get("raysPerPoint"));
    process.setProcessDuration(params.get("processTime"));

    domain->saveSurfaceMesh("initial.vtp");

    process.apply();

    domain->saveSurfaceMesh("final.vtp");

    // Remove mask
    domain->removeTopLevelSet();

    for (int i = 0; i < extraLayers.size(); ++i)
      domain->insertNextLevelSetAsMaterial(extraLayers[i], extraMaterials[i],
                                           false);
  }

  Scalar diodeHeight = params.get("bulkHeight") - params.get("holeDepth");
  Scalar solidHeight =
      params.get("bulkHeight") + params.get("passivationHeight") +
      params.get("antiReflectHeight1") + params.get("antiReflectHeight2");
  Scalar height =
      solidHeight + params.get("airHeight") + params.get("gridDelta");

  auto levelSets = domain->getLevelSets();
  auto materialMap = domain->getMaterialMap();

  Scalar gridDelta = std::min(params.get("gridDelta"),
                              params.get("fdtdWavelength") /
                                  params.get("fdtdCellsPerWavelength"));

  if (gridDelta < params.get("gridDelta"))
    levelSets = changeGridSpacing(levelSets, gridDelta);

  auto cellSet = cs::SmartPointer<cs::DenseCellSet<Scalar, D>>::New();
  cellSet->setCellSetPosition(/*isAboveSurface*/ true);
  cellSet->setCoverMaterial(int(ps::Material::Air));
  cellSet->fromLevelSets(
      levelSets, materialMap ? materialMap->getMaterialMap() : nullptr, height);
  cellSet->updateMaterials();
  cellSet->writeVTU("initial.vtu");

  // domain->generateCellSet(-5.0, ps::Material::Air, /*isAboveSurface*/ true)
  // auto& cellSet = domain->getCellSet();

  // Add lens
  using namespace fidi;
  Vec<D, int> csGridSize = getGridSize(*cellSet);
  Rect<D, Scalar> lensBounds(getBounds(*cellSet).size());
  lensBounds.min[D - 1] = solidHeight;
  lensBounds.max[D - 1] =
      lensBounds.min[D - 1] + params.get("airHeight") + params.get("gridDelta");
  setSphereMaterial(*cellSet->getScalarData("Material"), csGridSize, lensBounds,
                    project(lensBounds.max / Scalar(2), D - 1), height,
                    gridDelta, int(lensMaterial), int(ps::Material::Air));

  fdtd::MaterialMap matMap;
  matMap.emplace(int(ps::Material::Air), fdtd::Material{1, 1});
  matMap.emplace(int(ps::Material::Si),
                 fdtd::Material{params.get("siliconPermittivity"), 1});
  matMap.emplace(int(lensMaterial),
                 fdtd::Material{params.get("lensPermittivity"), 1});
  matMap.emplace(int(passivationMaterial),
                 fdtd::Material{params.get("passivationPermittivity"), 1});
  matMap.emplace(int(antiReflectMaterial1),
                 fdtd::Material{params.get("antiReflectPermittivity1"), 1});
  matMap.emplace(int(antiReflectMaterial2),
                 fdtd::Material{params.get("antiReflectPermittivity2"), 1});

  if (!params.get("fdtdWithMaterial"))
    for (auto &pair : matMap)
      pair.second.rA = pair.second.rB = 1;

  // Femtoseconds -> Microseconds
  auto fdtdOutputInterval = params.get("fdtdOutputInterval") / 1e9;
  auto fdtdTime = params.get("fdtdTime") / 1e9;

  runFDTD(*cellSet, std::move(matMap), fdtdTime, fdtdOutputInterval,
          params.get("fdtdWavelength"), diodeHeight);
  std::cout << "FDTD done\n";

  cellSet->writeVTU("final.vtu");
}
