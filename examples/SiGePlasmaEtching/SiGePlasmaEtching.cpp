#include <models/psCF4O2Etching.hpp>
#include <models/psDirectionalEtching.hpp>

#include <psDomain.hpp>
#include <psProcess.hpp>
#include <psUtil.hpp>

#include <lsCompareSparseField.hpp>
#include <lsVTKWriter.hpp>
#include <lsMesh.hpp>

#include "Geometry.hpp"
#include "Parameters.hpp"
#include "ReadPath.hpp"

namespace ps = viennaps;
namespace ls = viennals;

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 2;

  ps::Logger::setLogLevel(ps::LogLevel::WARNING);
  omp_set_num_threads(12);

  // Parse the parameters
  Parameters params;
  if (argc > 1) {
    std::cout << "Reading config file: " << argv[1] << std::endl;
    auto config = ps::util::readFile(argv[1]);
    if (config.empty()) {
      std::cerr << "Empty config provided" << std::endl;
      return -1;
    }
    params.fromMap(config);
  }

  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
  MakeInitialGeometry(geometry, params);
  geometry->saveVolumeMesh(params.fileName + "Initial");
  auto target = ps::SmartPointer<ps::Domain<NumericType, D>>::New(geometry);

  // directional etching of trenches
  if (params.pathFile.empty()) {
    std::cout << "Etching trenches ... " << std::endl;
    std::array<NumericType, 3> direction = {0., -1., 0.};
    NumericType isoVel =
        -0.5 * (params.trenchWidth - params.trenchWidthBot) /
        (params.numLayers * params.layerHeight + params.maskHeight);
    auto processModel =
        ps::SmartPointer<ps::DirectionalEtching<NumericType, D>>::New(
            direction, 1., isoVel, ps::Material::Mask);

    NumericType time = params.numLayers * params.layerHeight + params.overEtch +
                       params.maskHeight;
    ps::Process<NumericType, D>(geometry, processModel, time).apply();

    // remove trench etching mask (keep SiO2 mask)
    geometry->removeTopLevelSet();

  } else {
    // read geometry from file
    std::cout << "Importing " << params.pathFile << std::endl;
    ReadPath(geometry, target, params);
  }

  target->saveSurfaceMesh(params.fileName + "Target.vtp");
  if (params.saveVolume) {
    geometry->saveVolumeMesh(params.fileName + "Start");
    target->saveVolumeMesh(params.fileName + "Target");
  }

  // Parse the parameters
  ps::util::Parameters procParams;
  if (argc > 1) {
    procParams.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  // set parameter units
  ps::units::Length::setUnit(procParams.get<std::string>("lengthUnit"));
  ps::units::Time::setUnit(procParams.get<std::string>("timeUnit"));

  ps::CF4O2Parameters<NumericType> modelParams;
  modelParams.ionFlux = procParams.get("ionFlux");
  modelParams.etchantFlux = procParams.get("etchantFlux");
  modelParams.oxygenFlux = procParams.get("oxygenFlux");
  modelParams.polymerFlux = procParams.get("polymerFlux");
  modelParams.Ions.meanEnergy = procParams.get("meanEnergy");
  modelParams.Ions.sigmaEnergy = procParams.get("sigmaEnergy");
  modelParams.Passivation.A_O_ie = procParams.get("A_O");
  modelParams.Passivation.A_C_ie = procParams.get("A_C");
  auto model =
      ps::SmartPointer<ps::CF4O2Etching<NumericType, D>>::New(modelParams);

  // Process setup
  ps::Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setMaxCoverageInitIterations(10);
  process.setCoverageDeltaThreshold(1e-4);
  process.setNumberOfRaysPerPoint(
      static_cast<int>(procParams.get("numRaysPerPoint")));
  process.setProcessDuration(procParams.get("processTime"));
  process.setIntegrationScheme(ps::util::convertIntegrationScheme(
      procParams.get<std::string>("integrationScheme")));
  // process.setTimeStepRatio(0.4);

  std::cout << "Etching with CF4/O2 plasma ... " << std::endl;
  geometry->saveSurfaceMesh("etched_0.vtp", true);
  for (int i = 1; i <= procParams.get("numCycles"); ++i) {
    process.apply();
    geometry->saveSurfaceMesh("etched_" + std::to_string(i) + ".vtp", true);
  }

  geometry->saveVolumeMesh(params.fileName + "Final");

  auto targetls = ls::SmartPointer<ls::Domain<NumericType, D>>(target->getLevelSets().back());
  auto meshtarget = ls::SmartPointer<ls::Mesh<NumericType>>::New();
  ls::ToMesh(targetls, meshtarget).apply();
  ls::VTKWriter(meshtarget, "target.vtp").apply();

  auto geometryls = ls::SmartPointer<ls::Domain<NumericType, D>>(geometry->getLevelSets().back());
  auto meshgeom = ls::SmartPointer<ls::Mesh<NumericType>>::New();
  ls::ToMesh(geometryls, meshgeom).apply();
  ls::VTKWriter(meshgeom, "geometry.vtp").apply();

  ls::CompareSparseField<NumericType, D> compare(
    geometry->getLevelSets().back(), target->getLevelSets().back());
  
  auto comparisonMesh = ls::SmartPointer<ls::Mesh<NumericType>>::New();
  compare.setOutputMesh(comparisonMesh);
      
  compare.apply();
  ls::VTKWriter<NumericType>(comparisonMesh, "comparison.vtp").apply();

  compare.apply();
  std::cout << "sumSqDiff: " << compare.getSumSquaredDifferences() << std::endl;
  std::cout << "sumDiff: " << compare.getSumDifferences() << std::endl;
}