#include <geometries/psMakeHole.hpp>
#include <psProcess.hpp>
#include <psUtil.hpp>

#include <models/psCSVFileProcess.hpp>

using namespace viennaps;
constexpr int D = 3;
using NumericType = double;

#include <psMaterials.hpp>
#include <psVelocityField.hpp>

void runDeposition(SmartPointer<CSVFileProcess<NumericType, D>> &depoModel,
                   SmartPointer<Domain<NumericType, D>> &domain,
                   util::Parameters &params) {
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
    std::cout << "Usage: " << argv[0] << " <config file> [visualize]"
              << std::endl;
    return 1;
  }

  // Optional visualization
  if ((argc >= 3 && std::string(argv[2]) == "visualize")) {
    std::string cmd = std::string("python3 visualizeDomain.py ") + argv[1];
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
      std::cerr << "Visualization script failed or not found." << std::endl;
    }
  }

  // geometry setup
  auto geometry = SmartPointer<Domain<NumericType, D>>::New();

  MakeHole<NumericType, D>(geometry, params.get("gridDelta"),
                           params.get("xExtent"), params.get("yExtent"),
                           params.get("holeRadius"), params.get("holeDepth"),
                           params.get("taperingAngle"), 0.0, /* baseHeight */
                           false,          /* periodicBoundary */
                           false,          /* makeMask */
                           Material::Si,   /* material */
                           HoleShape::Full /* HoleShape */
                           )
      .apply();

  geometry->saveVolumeMesh("Hole");
  geometry->duplicateTopLevelSet(Material::SiO2);

  auto direction = Vec3D<NumericType>{0., 0., -1.};

  std::string ratesFile = params.get<std::string>("ratesFile");
  std::array<NumericType, D - 1> offset;
  offset[0] = params.get<NumericType>("offsetX");
  offset[1] = params.get<NumericType>("offsetY");

  auto depoModel = SmartPointer<CSVFileProcess<NumericType, D>>::New(
      ratesFile, direction, offset);

  std::string interpModeStr = params.get<std::string>("interpolationMode");
  if (interpModeStr == "linear") {
    depoModel->setInterpolationMode(
        impl::velocityFieldFromFile<NumericType, D>::Interpolation::LINEAR);
  } else if (interpModeStr == "idw") {
    depoModel->setInterpolationMode(
        impl::velocityFieldFromFile<NumericType, D>::Interpolation::IDW);
  } else if (interpModeStr == "custom") {
    // Define a custom interpolation function
    auto customInterp = [](const Vec3D<NumericType> &coord) -> NumericType {
      const NumericType x = coord[0];
      const NumericType y = coord[1];
      return 0.05 + 0.02 * std::sin(1.0 * x) * std::cos(1.0 * y);
    };
    depoModel->setCustomInterpolator(customInterp);
    depoModel->setInterpolationMode(
        impl::velocityFieldFromFile<NumericType, D>::Interpolation::CUSTOM);
  } else {
    std::cerr << "Warning: Unknown interpolation mode \"" << interpModeStr
              << "\" — using auto-detection." << std::endl;
  }

  const int numCycles = params.get<int>("numCycles");
  const std::string name = "HoleDeposition_";

  int n = 0;
  geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");

  for (int i = 0; i < numCycles; ++i) {
    std::cout << "Cycle " << i + 1 << std::endl;
    runDeposition(depoModel, geometry, params);
    geometry->saveSurfaceMesh(name + std::to_string(n++) + ".vtp");
  }

  geometry->saveVolumeMesh("HoleFinal");
}