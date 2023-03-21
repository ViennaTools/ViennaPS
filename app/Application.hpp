#pragma once

#include <fstream>
#include <sstream>

#include <lsReader.hpp>
#include <lsRemoveStrayPoints.hpp>

#include <psDomain.hpp>
#include <psGDSReader.hpp>
#include <psMakeHole.hpp>
#include <psMakePlane.hpp>
#include <psMakeTrench.hpp>
#include <psPlanarize.hpp>
#include <psProcess.hpp>
#include <psUtils.hpp>

#include <ApplicationParameters.hpp>
#include <ApplicationParser.hpp>

#include <DirectionalEtching.hpp>
#include <GeometricDistributionModels.hpp>
#include <IsotropicProcess.hpp>
#include <SF6O2Etching.hpp>
#include <SimpleDeposition.hpp>
#include <WetEtching.hpp>

template <int D> class Application {
  psSmartPointer<psDomain<NumericType, D>> geometry = nullptr;
  psSmartPointer<ApplicationParameters> params = nullptr;
  ApplicationParser parser;
  int clArgC = 0;
  char **clArgV;

public:
  Application(int argc, char **argv) : clArgC(argc), clArgV(argv) {}

  void run() {
    if (clArgC < 2) {
      std::cout << "No input file specified. " << std::endl;
      return;
    }

    std::fstream inputFile;
    inputFile.open(clArgV[1], std::fstream::in);

    if (!inputFile.is_open()) {
      std::cout << "Could not open input file." << std::endl;
      return;
    }

    params = psSmartPointer<ApplicationParameters>::New();
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
        writeVTP();
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

protected:
  virtual void
  runSimpleDeposition(psSmartPointer<psDomain<NumericType, D>> processGeometry,
                      psSmartPointer<ApplicationParameters> processParams) {
    // copy top layer for deposition
    auto depoLayer = psSmartPointer<lsDomain<NumericType, D>>::New(
        processGeometry->getLevelSets()->back());
    processGeometry->insertNextLevelSet(depoLayer);

    SimpleDeposition<NumericType, D> model(processParams->sticking,
                                           processParams->cosinePower);

    psProcess<NumericType, D> process;
    process.setDomain(processGeometry);
    process.setProcessModel(model.getProcessModel());
    process.setNumberOfRaysPerPoint(processParams->raysPerPoint);
    process.setProcessDuration(processParams->processTime *
                               processParams->rate / processParams->sticking);
    process.setPrintIntdermediate(params->printIntermediate);
    process.setIntegrationScheme(params->integrationScheme);
    process.apply();
  }

  virtual void
  runSF6O2Etching(psSmartPointer<psDomain<NumericType, D>> processGeometry,
                  psSmartPointer<ApplicationParameters> processParams) {
    SF6O2Etching<NumericType, D> model(
        processParams->totalIonFlux, processParams->totalEtchantFlux,
        processParams->totalOxygenFlux, processParams->ionEnergy,
        processParams->A_O, processParams->maskId);

    psProcess<NumericType, D> process;
    process.setDomain(processGeometry);
    process.setProcessModel(model.getProcessModel());
    process.setMaxCoverageInitIterations(10);
    process.setNumberOfRaysPerPoint(processParams->raysPerPoint);
    process.setProcessDuration(processParams->processTime);
    process.setPrintIntdermediate(params->printIntermediate);
    process.setIntegrationScheme(params->integrationScheme);
    process.apply();
  }

  virtual void runSphereDistribution(
      psSmartPointer<psDomain<NumericType, D>> processGeometry,
      psSmartPointer<ApplicationParameters> processParams) {
    SphereDistribution<NumericType, D> model(processParams->radius,
                                             processParams->gridDelta);

    psProcess<NumericType, D> process;
    process.setDomain(processGeometry);
    process.setProcessModel(model.getProcessModel());
    process.setPrintIntdermediate(params->printIntermediate);
    process.setIntegrationScheme(params->integrationScheme);
    process.apply();
  }

  virtual void
  runBoxDistribution(psSmartPointer<psDomain<NumericType, D>> processGeometry,
                     psSmartPointer<ApplicationParameters> processParams) {
    BoxDistribution<NumericType, D> model(processParams->halfAxes,
                                          processParams->gridDelta);

    psProcess<NumericType, D> process;
    process.setDomain(processGeometry);
    process.setProcessModel(model.getProcessModel());
    process.setPrintIntdermediate(params->printIntermediate);
    process.setIntegrationScheme(params->integrationScheme);
    process.apply();
  }

  virtual void runDirectionalEtching(
      psSmartPointer<psDomain<NumericType, D>> processGeometry,
      psSmartPointer<ApplicationParameters> processParams) {

    DirectionalEtching<NumericType, D> model(
        getDirection(processParams->direction), processParams->directionalRate,
        processParams->isotropicRate, processParams->maskId);

    psProcess<NumericType, D> process;
    process.setDomain(processGeometry);
    process.setProcessModel(model.getProcessModel());
    process.setProcessDuration(params->processTime);
    process.setPrintIntdermediate(params->printIntermediate);
    process.setIntegrationScheme(params->integrationScheme);
    process.apply();
  }

  virtual void
  runIsotropicProcess(psSmartPointer<psDomain<NumericType, D>> processGeometry,
                      psSmartPointer<ApplicationParameters> processParams) {

    if (params->rate > 0.) {
      // copy top layer for deposition
      auto depoLayer = psSmartPointer<lsDomain<NumericType, D>>::New(
          processGeometry->getLevelSets()->back());
      processGeometry->insertNextLevelSet(depoLayer);
    }

    IsotropicProcess<NumericType, D> model(processParams->rate,
                                           processParams->maskId);

    psProcess<NumericType, D> process;
    process.setDomain(processGeometry);
    process.setProcessModel(model.getProcessModel());
    process.setProcessDuration(params->processTime);
    process.setPrintIntdermediate(params->printIntermediate);
    process.setIntegrationScheme(params->integrationScheme);
    process.apply();
  }

  virtual void
  runWetEtching(psSmartPointer<psDomain<NumericType, D>> processGeometry,
                psSmartPointer<ApplicationParameters> processParams) {

    if constexpr (D == 3) {

      WetEtching<NumericType, D> model(processParams->maskId);
      psProcess<NumericType, D> process;
      process.setDomain(processGeometry);
      process.setProcessModel(model.getProcessModel());
      process.setProcessDuration(params->processTime);
      process.setIntegrationScheme(
          lsIntegrationSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER);
      process.setPrintIntdermediate(params->printIntermediate);
      process.apply();
    } else {
      std::cout << "Warning: Wet etch model only implemented in 3D."
                << std::endl;
    }
  }

private:
  void runInit() {
    std::cout << "\tx-Extent: " << params->xExtent
              << "\n\ty-Extent: " << params->yExtent
              << "\n\tResolution: " << params->gridDelta
              << "\n\tPrint intermediate: "
              << boolString(params->printIntermediate)
              << "\n\tPeriodic boundary: "
              << boolString(params->periodicBoundary)
              << "\n\tUsing integration scheme: "
              << intSchemeString(params->integrationScheme) << "\n\n";

    geometry = psSmartPointer<psDomain<NumericType, D>>::New();
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
      psMakeTrench<NumericType, D>(
          geometry, params->gridDelta, params->xExtent, params->yExtent,
          params->trenchWidth, params->trenchHeight, params->taperAngle,
          params->maskZPos, params->periodicBoundary, params->mask)
          .apply();
      break;

    case GeometryType::HOLE:
      std::cout << "Hole\n\tRadius: " << params->holeRadius
                << "\n\tDepth: " << params->holeDepth
                << "\n\tzPos: " << params->maskZPos
                << "\n\tTapering angle: " << params->taperAngle
                << "\n\tMask: " << boolString(params->mask) << "\n\n";
      psMakeHole<NumericType, D>(
          geometry, params->gridDelta, params->xExtent, params->yExtent,
          params->holeRadius, params->holeDepth, params->taperAngle,
          params->maskZPos, params->periodicBoundary, params->mask)
          .apply();
      break;

    case GeometryType::PLANE:
      std::cout << "Plane"
                << "\n\tzPos: " << params->maskZPos << "\n\n";
      if (geometry->getLevelSets()->back()) {
        std::cout << "\tAdding plane to current geometry...\n\n";
        psMakePlane<NumericType, D>(geometry, params->maskZPos, true).apply();
      } else {
        psMakePlane<NumericType, D>(geometry, params->gridDelta,
                                    params->xExtent, params->yExtent,
                                    params->maskZPos, params->periodicBoundary)
            .apply();
      }
      break;
    case GeometryType::GDS: {
      std::cout << "GDS file import\n\tFile name: " << params->fileName
                << "\n\tLayer: " << params->layers
                << "\n\tMask height: " << params->maskHeight
                << "\n\tzPos: " << params->maskZPos
                << "\n\tinvert: " << boolString(params->maskInvert)
                << "\n\txPadding: " << params->xPadding
                << "\n\tyPadding: " << params->yPadding << "\n\tPoint order: "
                << ((params->pointOrder == 0) ? "counter-clockwise"
                                              : "clockwise")
                << "\n\n";

      if constexpr (D == 3) {
        typename lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
        for (int i = 0; i < D - 1; i++) {
          if (params->periodicBoundary) {
            boundaryCons[i] =
                lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY;
          } else {
            boundaryCons[i] =
                lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
          }
        }
        boundaryCons[D - 1] =
            lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;
        auto mask = psSmartPointer<psGDSGeometry<NumericType, D>>::New(
            params->gridDelta);
        mask->setBoundaryConditions(boundaryCons);
        mask->setBoundaryPadding(params->xPadding, params->yPadding);
        if (params->pointOrder == 1) {
          mask->setPointOrder(psPointOrder::CLOCKWISE);
        }
        psGDSReader<NumericType, D>(mask, params->fileName).apply();

        auto layer =
            mask->layerToLevelSet(params->layers, params->maskZPos,
                                  params->maskHeight, params->maskInvert);
        geometry->insertNextLevelSet(layer);
      } else {
        std::cout << "Warning: Can only parse GDS geometries in 3D application."
                  << std::endl;
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
        auto layer = psSmartPointer<lsDomain<NumericType, D>>::New();
        lsReader<NumericType, D>(layer, layerFileName).apply();
        if (!geometry->getLevelSets()->empty() &&
            layer->getGrid().getGridDelta() !=
                geometry->getLevelSets()->back()->getGrid().getGridDelta()) {
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
    if (geometry->getLevelSets()->empty()) {
      std::cout << "Cannot run process on empty geometry." << std::endl;
      return;
    }

    std::cout << "\tModel: ";
    switch (params->processType) {
    case ProcessType::DEPOSITION: {
      std::cout << "Single particle deposition\n\tRate: " << params->rate
                << "\n\tTime: " << params->processTime
                << "\n\tSticking probability: " << params->sticking
                << "\n\tCosine exponent: " << params->cosinePower
                << "\n\tUsing " << params->raysPerPoint
                << " rays per source grid point\n\n";

      // copy top layer to capture deposition
      auto topLayerCopy = psSmartPointer<lsDomain<NumericType, D>>::New(
          geometry->getLevelSets()->back());
      geometry->insertNextLevelSet(topLayerCopy);
      runSimpleDeposition(geometry, params);
      break;
    }

    case ProcessType::SF6O2ETCHING: {
      std::cout << "SF6O2 etching\n\tTime: " << params->processTime
                << "\n\tEtchant flux: " << params->totalEtchantFlux
                << "\n\tOxygen flux: " << params->totalOxygenFlux
                << "\n\tIon flux: " << params->totalIonFlux
                << "\n\tIon energy: " << params->ionEnergy
                << "\n\tA_O: " << params->A_O << "\n\tUsing "
                << params->raysPerPoint << " rays per source grid point\n\n";
      runSF6O2Etching(geometry, params);
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

    case ProcessType::WETETCHING: {
      std::cout << "Wet etching\n\tTime: " << params->processTime
                << "\n\tUsing integration scheme: "
                << intSchemeString(lsIntegrationSchemeEnum::
                                       STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER)
                << "\n\n";
      runWetEtching(geometry, params);
      break;
    }

    case ProcessType::NONE:
      std::cout << "Process model could not be parsed. Skipping line."
                << std::endl;
      break;

    default:
      assert(false);
      break;
    }
  }

  void planarizeGeometry() {
    psPlanarize<NumericType, D>(geometry, params->maskZPos).apply();
  }

  void writeVTP() {
    if (geometry->getLevelSets()->empty()) {
      std::cout << "Cannot write empty geometry." << std::endl;
      return;
    }
    std::cout << "\tOut file name: " << params->fileName << ".vtp\n\n";
    geometry->printSurface(params->fileName + ".vtp");
  }

  std::array<NumericType, 3> getDirection(const std::string &directionString) {
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
      std::cout << "Invalid direction: " << directionString << std::endl;
    }

    return direction;
  }

  std::string boolString(const int in) { return in == 0 ? "false" : "true"; }

  std::string intSchemeString(lsIntegrationSchemeEnum scheme) {
    switch (scheme) {
    case lsIntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER:
      return "Enquist-Osher 1st Order";
    case lsIntegrationSchemeEnum::ENGQUIST_OSHER_2ND_ORDER:
      return "Enquist-Osher 2nd Order";
    case lsIntegrationSchemeEnum::LOCAL_LAX_FRIEDRICHS_1ST_ORDER:
      return "Local Lax-Friedrichs 1st Order";
    case lsIntegrationSchemeEnum::LOCAL_LAX_FRIEDRICHS_2ND_ORDER:
      return "Local Lax-Friedrichs 2nd Order";
    case lsIntegrationSchemeEnum::LAX_FRIEDRICHS_1ST_ORDER:
      return "Lax-Friedrichs 1st Order";
    case lsIntegrationSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER:
      return "Lax-Friedrichs 2nd Order";
    case lsIntegrationSchemeEnum::LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER:
      return "Local Lax-Friedrichs Analytical 1st Order";
    case lsIntegrationSchemeEnum::LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER:
      return "Local Local Lax-Friedrichs 1st Order";
    case lsIntegrationSchemeEnum::LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER:
      return "Local Local Lax-Friedrichs 2nd Order";
    case lsIntegrationSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER:
      return "Stencil Lax-Friedrichs 1st Order";
    }

    return "Invalid integration scheme";
  }
};