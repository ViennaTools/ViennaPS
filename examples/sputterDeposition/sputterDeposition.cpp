#include <geometries/psMakeTrench.hpp>
#include <psProcess.hpp>
#include <psUtil.hpp>

#include <models/psCSVFileProcess.hpp>

using namespace viennaps;
constexpr int D = 2;
using NumericType = double;

#include <psMaterials.hpp>
#include <psVelocityField.hpp>
#include <vector>

void runDeposition(SmartPointer<CSVFileProcess<NumericType, D>> &depoModel,
                   SmartPointer<Domain<NumericType, D>> &domain,
                   util::Parameters &params) {
  std::cout << "  - Depositing - " << std::endl;
  Process<NumericType, D>(domain, depoModel, params.get("depositionTime"))
      .apply();
}

int main(int argc, char **argv) {

  Logger::setLogLevel(LogLevel::ERROR);
  omp_set_num_threads(16);

  // Parse the parameters
  util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  // geometry setup
  auto geometry = SmartPointer<Domain<NumericType, D>>::New();

  MakeTrench<NumericType, D>(geometry, params.get("gridDelta"),
                             params.get("xExtent"), params.get("yExtent"),
                             params.get("trenchWidth"),
                             params.get("trenchDepth"),
                             params.get("taperingAngle"), 0.0, /* baseHeight */
                             false,       /* periodicBoundary */
                             false,       /* makeMask */
                             Material::Si /* material */
                             )
      .apply();

  geometry->saveVolumeMesh("Trench");
  geometry->duplicateTopLevelSet(Material::SiO2);

  auto direction = Vec3D<NumericType>{0., -1., 0.};

  std::string ratesFile = params.get<std::string>("ratesFile");
  std::array<NumericType, D - 1> offset;
  offset[0] = params.get<NumericType>("offsetX");

  auto depoModel = SmartPointer<CSVFileProcess<NumericType, D>>::New(
      ratesFile, direction, offset);

  const int numCycles = params.get<int>("numCycles");
  const std::string name = "TrenchDeposition_";

  int n = 0;
  geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");

  for (int i = 0; i < numCycles; ++i) {
    std::cout << "Cycle " << i + 1 << std::endl;
    runDeposition(depoModel, geometry, params);
    geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");
  }

  geometry->saveVolumeMesh("TrenchFinal");
}