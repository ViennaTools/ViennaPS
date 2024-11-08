#pragma once

#include <fstream>
#include <sstream>

#include <lsReader.hpp>

#include <geometries/psMakeHole.hpp>
#include <geometries/psMakePlane.hpp>
#include <geometries/psMakeStack.hpp>
#include <geometries/psMakeTrench.hpp>

#include <models/psAnisotropicProcess.hpp>
#include <models/psDirectionalEtching.hpp>
#include <models/psFluorocarbonEtching.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psSF6O2Etching.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <models/psTEOSDeposition.hpp>

#include <psDomain.hpp>
#include <psGDSReader.hpp>
#include <psPlanarize.hpp>
#include <psProcess.hpp>
#include <psUtils.hpp>

#include "applicationParameters.hpp"
#include "applicationParser.hpp"

using namespace viennaps;

template <int D> class Application {
  SmartPointer<Domain<NumericType, D>> geometry = nullptr;
  SmartPointer<ApplicationParameters> params = nullptr;
  ApplicationParser parser;
  int clArgC = 0;
  char **clArgV;

public:
  Application(int argc, char **argv) : clArgC(argc), clArgV(argv) {}

  void run() {
    if (clArgC < 2) {
      Logger::getInstance().addError("No input file specified.").print();
      return;
    }

    std::fstream inputFile;
    inputFile.open(clArgV[1], std::fstream::in);

    if (!inputFile.is_open()) {
      Logger::getInstance().addError("Could not open input file.").print();
      return;
    }

    params = SmartPointer<ApplicationParameters>::New();
    params->defaultParameters(true);
    parser.setParameters(params);

    int lineNumber = 0;
    std::string line;
    while (std::getline(inputFile, line)) {
      if (line.empty() || line[0] == '#') // skipping empty line and comments
        continue;
      std::istringstream lineStream(line);
      switch (parser.parseCommand(lineStream, lineNumber)) {
      case CommandType::INIT:
        runInit();
        break;

      case CommandType::GEOMETRY:
        createGeometry();
        break;

      case CommandType::PROCESS:
        runProcess();
        break;

      case CommandType::PLANARIZE:
        planarizeGeometry();
        break;

      case CommandType::OUTPUT:
        writeOutput();
        break;

      case CommandType::NONE:
        break;

      default:
        assert(false);
      }

      lineNumber++;
      params->defaultParameters();
    }
  }

  void printGeometry(std::string fileName) {
    params->fileName = fileName;
    writeOutput();
  }

protected:
  virtual void
  runSingleParticleProcess(SmartPointer<Domain<NumericType, D>> processGeometry,
                           SmartPointer<ApplicationParameters> processParams) {

    // copy top layer for deposition
    processGeometry->duplicateTopLevelSet(processParams->material);

    auto model = SmartPointer<SingleParticleProcess<NumericType, D>>::New(
        processParams->rate, processParams->sticking,
        processParams->cosinePower);

    Process<NumericType, D> process;
    process.setDomain(processGeometry);
    process.setProcessModel(model);
    if (processParams->smoothFlux)
      process.enableFluxSmoothing();
    process.setNumberOfRaysPerPoint(processParams->raysPerPoint);
    process.setProcessDuration(processParams->processTime);
    process.setIntegrationScheme(params->integrationScheme);
    process.apply();
  }

  virtual void
  runTEOSDeposition(SmartPointer<Domain<NumericType, D>> processGeometry,
                    SmartPointer<ApplicationParameters> processParams) {
    // copy top layer for deposition
    processGeometry->duplicateTopLevelSet(processParams->material);

    auto model = SmartPointer<TEOSDeposition<NumericType, D>>::New(
        processParams->stickingP1, processParams->rateP1,
        processParams->orderP1, processParams->stickingP2,
        processParams->rateP2, processParams->orderP2);

    Process<NumericType, D> process;
    process.setDomain(processGeometry);
    process.setProcessModel(model);
    if (processParams->smoothFlux)
      process.enableFluxSmoothing();
    process.setNumberOfRaysPerPoint(processParams->raysPerPoint);
    process.setProcessDuration(processParams->processTime);
    process.setIntegrationScheme(params->integrationScheme);
    process.apply();
  }

  virtual void
  runSF6O2Etching(SmartPointer<Domain<NumericType, D>> processGeometry,
                  SmartPointer<ApplicationParameters> processParams) {
    auto model = SmartPointer<SF6O2Etching<NumericType, D>>::New(
        processParams->ionFlux, processParams->etchantFlux,
        processParams->oxygenFlux, processParams->ionEnergy,
        processParams->sigmaIonEnergy, processParams->ionExponent,
        processParams->A_O);

    Process<NumericType, D> process;
    process.setDomain(processGeometry);
    process.setProcessModel(model);
    process.setMaxCoverageInitIterations(10);
    if (processParams->smoothFlux)
      process.enableFluxSmoothing();
    process.setNumberOfRaysPerPoint(processParams->raysPerPoint);
    process.setProcessDuration(processParams->processTime);
    process.setIntegrationScheme(params->integrationScheme);
    process.apply();
  }

  virtual void
  runFluorocarbonEtching(SmartPointer<Domain<NumericType, D>> processGeometry,
                         SmartPointer<ApplicationParameters> processParams) {
    auto model = SmartPointer<FluorocarbonEtching<NumericType, D>>::New(
        processParams->ionFlux, processParams->etchantFlux,
        processParams->oxygenFlux, processParams->ionEnergy,
        processParams->sigmaIonEnergy, processParams->ionExponent,
        processParams->deltaP);

    Process<NumericType, D> process;
    process.setDomain(processGeometry);
    process.setProcessModel(model);
    process.setMaxCoverageInitIterations(10);
    if (processParams->smoothFlux)
      process.enableFluxSmoothing();
    process.setNumberOfRaysPerPoint(processParams->raysPerPoint);
    process.setProcessDuration(processParams->processTime);
    process.setIntegrationScheme(params->integrationScheme);
    process.apply();
  }

  virtual void
  runSphereDistribution(SmartPointer<Domain<NumericType, D>> processGeometry,
                        SmartPointer<ApplicationParameters> processParams) {
    auto model = SmartPointer<SphereDistribution<NumericType, D>>::New(
        processParams->radius, processParams->gridDelta);

    Process<NumericType, D> process;
    process.setDomain(processGeometry);
    process.setProcessModel(model);
    process.setIntegrationScheme(params->integrationScheme);
    process.apply();
  }

  virtual void
  runBoxDistribution(SmartPointer<Domain<NumericType, D>> processGeometry,
                     SmartPointer<ApplicationParameters> processParams) {
    auto model = SmartPointer<BoxDistribution<NumericType, D>>::New(
        processParams->halfAxes, processParams->gridDelta);

    Process<NumericType, D> process;
    process.setDomain(processGeometry);
    process.setProcessModel(model);
    process.setIntegrationScheme(params->integrationScheme);
    process.apply();
  }

  virtual void
  runDirectionalEtching(SmartPointer<Domain<NumericType, D>> processGeometry,
                        SmartPointer<ApplicationParameters> processParams) {

    auto model = SmartPointer<DirectionalEtching<NumericType, D>>::New(
        getDirection(processParams->direction), processParams->directionalRate,
        processParams->isotropicRate, processParams->maskMaterial);

    Process<NumericType, D> process;
    process.setDomain(processGeometry);
    process.setProcessModel(model);
    process.setProcessDuration(params->processTime);
    process.setIntegrationScheme(params->integrationScheme);
    process.apply();
  }

  virtual void
  runIsotropicProcess(SmartPointer<Domain<NumericType, D>> processGeometry,
                      SmartPointer<ApplicationParameters> processParams) {

    if (params->rate > 0.) {
      // copy top layer for deposition
      processGeometry->duplicateTopLevelSet(processParams->material);
    }

    auto model = SmartPointer<IsotropicProcess<NumericType, D>>::New(
        processParams->rate, processParams->maskMaterial);

    Process<NumericType, D> process;
    process.setDomain(processGeometry);
    process.setProcessModel(model);
    process.setProcessDuration(params->processTime);
    process.setIntegrationScheme(params->integrationScheme);
    process.apply();
  }

  virtual void
  runAnisotropicProcess(SmartPointer<Domain<NumericType, D>> processGeometry,
                        SmartPointer<ApplicationParameters> processParams) {
    Logger::getInstance()
        .addError("Warning: Anisotropic process model not implemented in "
                  "application.")
        .print();
  }

private:
  void runInit() {
    std::cout << "\tx-Extent: " << params->xExtent
              << "\n\ty-Extent: " << params->yExtent
              << "\n\tResolution: " << params->gridDelta
              << "\n\tLog level: " << params->logLevel
              << "\n\tPeriodic boundary: "
              << boolString(params->periodicBoundary)
              << "\n\tUsing integration scheme: "
              << intSchemeString(params->integrationScheme) << "\n\n";

    geometry = SmartPointer<Domain<NumericType, D>>::New();
  }

  void createGeometry() {
    std::cout << "\tGeometry type: ";
    switch (params->geometryType) {
    case GeometryType::TRENCH:
      std::cout << "Trench\n\tWidth: " << params->trenchWidth
                << "\n\tHeight: " << params->trenchHeight
                << "\n\tzPos: " << params->maskZPos
                << "\n\tTapering angle: " << params->taperAngle
                << "\n\tMask: " << boolString(params->mask) << "\n\n";
      MakeTrench<NumericType, D>(geometry, params->gridDelta, params->xExtent,
                                 params->yExtent, params->trenchWidth,
                                 params->trenchHeight, params->taperAngle,
                                 params->maskZPos, params->periodicBoundary,
                                 params->mask, params->material)
          .apply();
      break;

    case GeometryType::HOLE:
      std::cout << "Hole\n\tRadius: " << params->holeRadius
                << "\n\tDepth: " << params->holeDepth
                << "\n\tzPos: " << params->maskZPos
                << "\n\tTapering angle: " << params->taperAngle
                << "\n\tMask: " << boolString(params->mask) << "\n\n";
      MakeHole<NumericType, D>(geometry, params->gridDelta, params->xExtent,
                               params->yExtent, params->holeRadius,
                               params->holeDepth, params->taperAngle,
                               params->maskZPos, params->periodicBoundary,
                               params->mask, params->material)
          .apply();
      break;

    case GeometryType::PLANE:
      std::cout << "Plane"
                << "\n\tzPos: " << params->maskZPos << "\n\n";
      if (!geometry->getLevelSets().empty()) {
        std::cout << "\tAdding plane to current geometry...\n\n";
        MakePlane<NumericType, D>(geometry, params->maskZPos, params->material)
            .apply();
      } else {
        MakePlane<NumericType, D>(geometry, params->gridDelta, params->xExtent,
                                  params->yExtent, params->maskZPos,
                                  params->periodicBoundary, params->material)
            .apply();
      }
      break;

    case GeometryType::STACK:
      std::cout << "Stack\n\tNumber of layers: " << params->numLayers
                << "\n\tLayer height: " << params->layerHeight
                << "\n\tSubstrate height: " << params->substrateHeight
                << "\n\tHole radius: " << params->holeRadius
                << "\n\tMask height: " << params->maskHeight << "\n\n";
      MakeStack<NumericType, D>(
          geometry, params->gridDelta, params->xExtent, params->yExtent,
          params->numLayers, params->layerHeight, params->substrateHeight,
          params->holeRadius, params->maskHeight, params->periodicBoundary)
          .apply();

      break;

    case GeometryType::GDS: {
      std::cout << "GDS file import\n\tFile name: " << params->fileName
                << "\n\tLayer: " << params->layers
                << "\n\tMask height: " << params->maskHeight
                << "\n\tzPos: " << params->maskZPos
                << "\n\tinvert: " << boolString(params->maskInvert)
                << "\n\txPadding: " << params->xPadding
                << "\n\tyPadding: " << params->yPadding << "\n\tPoint order: "
                << "\n\n";

      if constexpr (D == 3) {
        typename viennals::Domain<NumericType, D>::BoundaryType boundaryCons[D];
        for (int i = 0; i < D - 1; i++) {
          if (params->periodicBoundary) {
            boundaryCons[i] =
                viennals::Domain<NumericType,
                                 D>::BoundaryType::PERIODIC_BOUNDARY;
          } else {
            boundaryCons[i] =
                viennals::Domain<NumericType,
                                 D>::BoundaryType::REFLECTIVE_BOUNDARY;
          }
        }
        boundaryCons[D - 1] =
            viennals::Domain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;
        auto mask =
            SmartPointer<GDSGeometry<NumericType, D>>::New(params->gridDelta);
        mask->setBoundaryConditions(boundaryCons);
        mask->setBoundaryPadding(params->xPadding, params->yPadding);
        GDSReader<NumericType, D>(mask, params->fileName).apply();

        auto layer =
            mask->layerToLevelSet(params->layers, params->maskZPos,
                                  params->maskHeight, params->maskInvert);
        geometry->insertNextLevelSetAsMaterial(layer, params->material);
      } else {
        Logger::getInstance()
            .addError("Can only parse GDS geometries in 3D application.")
            .print();
      }
      break;
    }

    case GeometryType::IMPORT: {
      std::cout << "ViennaLS file import\n\tFile name: " << params->fileName
                << "\n\tNumber of layers: " << params->layers << "\n";
      for (int i = 0; i < params->layers; i++) {
        std::string layerFileName =
            params->fileName + "_layer" + std::to_string(i) + ".lvst";
        std::cout << "\tReading " << layerFileName << std::endl;
        auto layer = SmartPointer<viennals::Domain<NumericType, D>>::New();
        viennals::Reader<NumericType, D>(layer, layerFileName).apply();
        if (!geometry->getLevelSets().empty() &&
            layer->getGrid().getGridDelta() !=
                geometry->getLevelSets().back()->getGrid().getGridDelta()) {
          std::cout << std::setprecision(8);
          std::cout << "Import geometry grid does not match. Grid resolution: "
                    << params->gridDelta << ", Import grid resolution: "
                    << layer->getGrid().getGridDelta()
                    << "\nCannot add geometry." << std::endl;
          continue;
        }
        geometry->insertNextLevelSet(layer, false);
      }
      break;
    }

    default:
      break;
    }
  }

  void runProcess() {
    if (geometry->getLevelSets().empty()) {
      Logger::getInstance()
          .addError("Cannot run process on empty geometry.")
          .print();
      return;
    }

    std::cout << "\tModel: ";
    switch (params->processType) {
    case ProcessType::SINGLEPARTICLEPROCESS: {
      std::cout << "Single particle deposition\n\tRate: " << params->rate
                << "\n\tTime: " << params->processTime
                << "\n\tMaterial: " << materialString(params->material)
                << "\n\tSticking probability: " << params->sticking
                << "\n\tCosine exponent: " << params->cosinePower
                << "\n\tUsing " << params->raysPerPoint
                << " rays per source grid point\n\n";
      runSingleParticleProcess(geometry, params);
      break;
    }

    case ProcessType::TEOSDEPOSITION: {
      if (params->rateP2 != 0.) {
        std::cout << "Multi particle TEOS deposition"
                  << "\n\tP1 rate: " << params->rateP1
                  << "\n\tP1 sticking probability: " << params->stickingP1
                  << "\n\tP2 reaction order: " << params->orderP1
                  << "\n\tP2 rate: " << params->rateP2
                  << "\n\tP2 sticking probability: " << params->stickingP2
                  << "\n\tP2 reaction order: " << params->orderP2
                  << "\n\tTime: " << params->processTime
                  << "\n\tMaterial: " << materialString(params->material)
                  << "\n\tUsing " << params->raysPerPoint
                  << " rays per source grid point\n\n";
      } else {
        std::cout << "Single particle TEOS deposition\n\tRate: "
                  << params->rateP1
                  << "\n\tSticking probability: " << params->stickingP1
                  << "\n\tReaction order: " << params->orderP1
                  << "\n\tTime: " << params->processTime
                  << "\n\tMaterial: " << materialString(params->material)
                  << "\n\tUsing " << params->raysPerPoint
                  << " rays per source grid point\n\n";
      }
      runTEOSDeposition(geometry, params);
      break;
    }

    case ProcessType::SF6O2ETCHING: {
      std::cout << "SF6O2 etching\n\tTime: " << params->processTime
                << "\n\tEtchant flux: " << params->etchantFlux
                << "\n\tOxygen flux: " << params->oxygenFlux
                << "\n\tIon flux: " << params->ionFlux
                << "\n\tIon energy: " << params->ionEnergy
                << "\n\tIon exponent: " << params->ionExponent
                << "\n\tA_O: " << params->A_O << "\n\tUsing "
                << params->raysPerPoint << " rays per source grid point\n\n";
      runSF6O2Etching(geometry, params);
      break;
    }

    case ProcessType::FLUOROCARBONETCHING: {
      std::cout << "Fluorocarbon etching\n\tTime: " << params->processTime
                << "\n\tEtchant flux: " << params->etchantFlux
                << "\n\tOxygen flux: " << params->oxygenFlux
                << "\n\tIon flux: " << params->ionFlux
                << "\n\tIon energy: " << params->ionEnergy
                << "\n\tDelta P: " << params->deltaP << "\n\tUsing "
                << params->raysPerPoint << " rays per source grid point\n\n";
      runFluorocarbonEtching(geometry, params);
      break;
    }

    case ProcessType::SPHEREDISTRIBUTION: {
      std::cout << "Sphere Distribution\n\tRadius: " << params->radius
                << "\n\n";
      runSphereDistribution(geometry, params);
      break;
    }

    case ProcessType::BOXDISTRIBUTION: {
      std::cout << "Box Distribution\n\thalfAxes: (" << params->halfAxes[0]
                << ',' << params->halfAxes[1] << ',' << params->halfAxes[2]
                << ")\n\n";
      runBoxDistribution(geometry, params);
      break;
    }

    case ProcessType::DIRECTIONALETCHING: {
      std::cout << "Directional etching\n\tTime: " << params->processTime
                << "\n\tDirectional rate: " << params->directionalRate
                << "\n\tIsotropic rate: " << params->isotropicRate << "\n\n";
      runDirectionalEtching(geometry, params);
      break;
    }

    case ProcessType::ISOTROPIC: {
      std::cout << "Isotropic process\n\tTime: " << params->processTime
                << "\n\tIsotropic rate: " << params->rate << "\n\n";
      runIsotropicProcess(geometry, params);
      break;
    }

    case ProcessType::ANISOTROPIC: {
      std::cout << "Wet etching\n\tTime: " << params->processTime
                << "\n\tUsing integration scheme: "
                << intSchemeString(viennals::IntegrationSchemeEnum::
                                       STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER)
                << "\n\n";
      runAnisotropicProcess(geometry, params);
      break;
    }

    case ProcessType::NONE:
      Logger::getInstance()
          .addWarning("Process model could not be parsed. Skipping line.")
          .print();
      break;

    default:
      assert(false);
      break;
    }
  }

  void planarizeGeometry() {
    Planarize<NumericType, D>(geometry, params->maskZPos).apply();
  }

  void writeOutput() {
    if (geometry->getLevelSets().empty()) {
      std::cout << "Cannot write empty geometry." << std::endl;
      return;
    }

    std::string outFileName = params->fileName;
    if (params->out == OutputType::SURFACE) {
      std::cout << "\tWriting surface ...\n";
      const std::string suffix = ".vtp";
      // check if string ends with .vtp
      if (!(outFileName.size() >= suffix.size() &&
            0 == outFileName.compare(outFileName.size() - suffix.size(),
                                     suffix.size(), suffix))) {
        outFileName += ".vtp";
      }
      geometry->saveSurfaceMesh(outFileName);
    } else {
      std::cout << "Writing volume ...\n";
      const std::string suffix = ".vtu";
      // check if string ends with .vtu
      if (outFileName.size() > suffix.size() &&
          std::equal(suffix.rbegin(), suffix.rend(), suffix.rbegin())) {
        outFileName.erase(outFileName.length() - 4);
      }
      geometry->saveVolumeMesh(outFileName);
      outFileName += "_volume.vtu";
    }
    std::cout << "\tOut file name: " << outFileName << "\n\n";
  }

  static std::array<NumericType, 3>
  getDirection(const std::string &directionString) {
    std::array<NumericType, 3> direction = {0};

    if (directionString == "negZ") {
      int i = 2;
      if constexpr (D == 2)
        i--;
      direction[i] = -1.;
    } else if (directionString == "posZ") {
      int i = 2;
      if constexpr (D == 2)
        i--;
      direction[i] = 1.;
    } else if (directionString == "negY") {
      direction[1] = -1.;
    } else if (directionString == "posY") {
      direction[1] = 1.;
    } else if (directionString == "negX") {
      direction[0] = -1.;
    } else if (directionString == "posX") {
      direction[0] = 1.;
    } else {
      Logger::getInstance()
          .addError("Invalid direction: " + directionString)
          .print();
    }

    return direction;
  }

  static inline std::string boolString(const int in) {
    return in == 0 ? "false" : "true";
  }

  static std::string materialString(const Material material) {
    switch (material) {
    case Material::None:
      return "None";
    case Material::Mask:
      return "Mask";
    case Material::Si:
      return "Si";
    case Material::Si3N4:
      return "Si3N4";
    case Material::SiO2:
      return "SiO2";
    case Material::SiON:
      return "SiON";
    case Material::PolySi:
      return "PolySi";
    case Material::Polymer:
      return "Polymer";
    case Material::SiC:
      return "SiC";
    case Material::SiN:
      return "SiN";
    case Material::Metal:
      return "Metal";
    case Material::W:
      return "W";
    case Material::TiN:
      return "TiN";
    case Material::GaN:
      return "GaN";
    case Material::GAS:
      return "GAS";
    case Material::Air:
      return "Air";
    case Material::Al2O3:
      return "Al2O3";
    case Material::Dielectric:
      return "Dielectric";
    case Material::Cu:
      return "Cu";
    default:
      return "Unknown material";
    }
  }

  std::string intSchemeString(viennals::IntegrationSchemeEnum scheme) {
    switch (scheme) {
    case viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER:
      return "Enquist-Osher 1st Order";
    case viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_2ND_ORDER:
      return "Enquist-Osher 2nd Order";
    case viennals::IntegrationSchemeEnum::LOCAL_LAX_FRIEDRICHS_1ST_ORDER:
      return "Local Lax-Friedrichs 1st Order";
    case viennals::IntegrationSchemeEnum::LOCAL_LAX_FRIEDRICHS_2ND_ORDER:
      return "Local Lax-Friedrichs 2nd Order";
    case viennals::IntegrationSchemeEnum::LAX_FRIEDRICHS_1ST_ORDER:
      return "Lax-Friedrichs 1st Order";
    case viennals::IntegrationSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER:
      return "Lax-Friedrichs 2nd Order";
    case viennals::IntegrationSchemeEnum::
        LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER:
      return "Local Lax-Friedrichs Analytical 1st Order";
    case viennals::IntegrationSchemeEnum::LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER:
      return "Local Local Lax-Friedrichs 1st Order";
    case viennals::IntegrationSchemeEnum::LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER:
      return "Local Local Lax-Friedrichs 2nd Order";
    case viennals::IntegrationSchemeEnum::
        STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER:
      return "Stencil Lax-Friedrichs 1st Order";
    }

    return "Invalid integration scheme";
  }
};
