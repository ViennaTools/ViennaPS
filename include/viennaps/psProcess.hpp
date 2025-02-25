#pragma once

#include "psProcessModel.hpp"
#include "psProcessParams.hpp"
#include "psTranslationField.hpp"
#include "psUnits.hpp"
#include "psUtils.hpp"

#include <lsAdvect.hpp>
#include <lsCalculateVisibilities.hpp>
#include <lsDomain.hpp>
#include <lsMesh.hpp>
#include <lsToDiskMesh.hpp>
#include <lsVTKWriter.hpp>

#include <rayParticle.hpp>
#include <rayTrace.hpp>

namespace viennaps {

using namespace viennacore;

/// This class server as the main process tool, applying a user- or pre-defined
/// process model to a domain. Depending on the user inputs surface advection, a
/// single callback function or a geometric advection is applied.
template <typename NumericType, int D> class Process {
  using translatorType = std::unordered_map<unsigned long, unsigned long>;
  using psDomainType = SmartPointer<Domain<NumericType, D>>;

public:
  Process() {}
  Process(psDomainType passedDomain) : domain(passedDomain) {}
  // Constructor for a process with a pre-configured process model.
  Process(psDomainType passedDomain,
          SmartPointer<ProcessModel<NumericType, D>> passedProcessModel,
          const NumericType passedDuration = 0.)
      : domain(passedDomain), model(passedProcessModel),
        processDuration(passedDuration) {}

  // Set the process model. This can be either a pre-configured process model or
  // a custom process model. A custom process model must interface the
  // psProcessModel class.
  void setProcessModel(
      SmartPointer<ProcessModel<NumericType, D>> passedProcessModel) {
    model = passedProcessModel;
  }

  // Set the process domain.
  void setDomain(psDomainType passedDomain) { domain = passedDomain; }

  /* ----- Process parameters ----- */

  // Set the duration of the process.
  void setProcessDuration(NumericType passedDuration) {
    processDuration = passedDuration;
  }

  // Returns the duration of the recently run process. This duration can
  // sometimes slightly vary from the set process duration, due to the maximum
  // time step according to the CFL condition.
  NumericType getProcessDuration() const { return processTime; }

  /* ----- Ray tracing parameters ----- */

  // Set the number of iterations to initialize the coverages.
  void setMaxCoverageInitIterations(unsigned maxIt) { maxIterations = maxIt; }

  // Set the threshold for the coverage delta metric to reach convergence.
  void setCoverageDeltaThreshold(NumericType treshold) {
    coverageDeltaThreshold = treshold;
  }

  // Set the source direction, where the rays should be traced from.
  void setSourceDirection(viennaray::TraceDirection passedDirection) {
    rayTracingParams.sourceDirection = passedDirection;
  }

  // Specify the number of rays to be traced for each particle throughout the
  // process. The total count of rays is the product of this number and the
  // number of points in the process geometry.
  void setNumberOfRaysPerPoint(unsigned numRays) {
    rayTracingParams.raysPerPoint = numRays;
  }

  // Enable flux smoothing. The flux at each surface point, calculated
  // by the ray tracer, is averaged over the surface point neighbors.
  void enableFluxSmoothing() { rayTracingParams.smoothingNeighbors = 1; }

  // Disable flux smoothing.
  void disableFluxSmoothing() { rayTracingParams.smoothingNeighbors = 0; }

  void setRayTracingDiskRadius(NumericType radius) {
    rayTracingParams.diskRadius = radius;
    if (rayTracingParams.diskRadius < 0.) {
      Logger::getInstance()
          .addWarning("Disk radius must be positive. Using default value.")
          .print();
      rayTracingParams.diskRadius = 0.;
    }
  }

  void enableFluxBoundaries() { rayTracingParams.ignoreFluxBoundaries = false; }

  // Ignore boundary conditions during the flux calculation.
  void disableFluxBoundaries() { rayTracingParams.ignoreFluxBoundaries = true; }

  // Enable the use of random seeds for ray tracing. This is useful to
  // prevent the formation of artifacts in the flux calculation.
  void enableRandomSeeds() { rayTracingParams.useRandomSeeds = true; }

  // Disable the use of random seeds for ray tracing.
  void disableRandomSeeds() { rayTracingParams.useRandomSeeds = false; }

  void setRayTracingParameters(
      const RayTracingParameters<NumericType, D> &passedRayTracingParams) {
    rayTracingParams = passedRayTracingParams;
  }

  auto &getRayTracingParameters() { return rayTracingParams; }

  /* ----- Advection parameters ----- */

  // Set the integration scheme for solving the level-set equation.
  // Possible integration schemes are specified in
  // viennals::IntegrationSchemeEnum.
  void setIntegrationScheme(IntegrationScheme passedIntegrationScheme) {
    advectionParams.integrationScheme = passedIntegrationScheme;
  }

  // Enable the output of the advection velocities on the level-set mesh.
  void enableAdvectionVelocityOutput() {
    advectionParams.velocityOutput = true;
  }

  // Disable the output of the advection velocities on the level-set mesh.
  void disableAdvectionVelocityOutput() {
    advectionParams.velocityOutput = false;
  }

  // Set the CFL (Courant-Friedrichs-Levy) condition to use during surface
  // advection in the level-set. The CFL condition defines the maximum distance
  // a surface is allowed to move in a single advection step. It MUST be below
  // 0.5 to guarantee numerical stability. Defaults to 0.4999.
  void setTimeStepRatio(NumericType cfl) {
    advectionParams.timeStepRatio = cfl;
  }

  void setAdvectionParameters(
      const AdvectionParameters<NumericType> &passedAdvectionParams) {
    advectionParams = passedAdvectionParams;
  }

  auto &getAdvectionParameters() { return advectionParams; }

  /* ----- Process execution ----- */

  // A single flux calculation is performed on the domain surface. The result is
  // stored as point data on the nodes of the mesh.
  SmartPointer<viennals::Mesh<NumericType>> calculateFlux() {
    model->initialize(domain, 0.);
    const auto name = model->getProcessName().value_or("default");
    const auto logLevel = Logger::getLogLevel();

    // Generate disk mesh from domain
    auto mesh = SmartPointer<viennals::Mesh<NumericType>>::New();
    viennals::ToDiskMesh<NumericType, D> meshConverter(mesh);
    for (auto dom : domain->getLevelSets()) {
      meshConverter.insertNextLevelSet(dom);
    }
    meshConverter.apply();

    viennaray::Trace<NumericType, D> rayTracer;
    initializeRayTracer(rayTracer);

    auto points = mesh->getNodes();
    auto normals = *mesh->getCellData().getVectorData("Normals");
    auto materialIds = *mesh->getCellData().getScalarData("MaterialIds");

    if (rayTracingParams.diskRadius == 0.) {
      rayTracer.setGeometry(points, normals, domain->getGrid().getGridDelta());
    } else {
      rayTracer.setGeometry(points, normals, domain->getGrid().getGridDelta(),
                            rayTracingParams.diskRadius);
    }
    rayTracer.setMaterialIds(materialIds);

    const bool useProcessParams =
        model->getSurfaceModel()->getProcessParameters() != nullptr;
    bool useCoverages = false;

    // Initialize coverages
    model->getSurfaceModel()->initializeSurfaceData(points.size());
    if (!coveragesInitialized_)
      model->getSurfaceModel()->initializeCoverages(points.size());
    auto coverages = model->getSurfaceModel()->getCoverages();
    std::ofstream covMetricFile;
    if (coverages != nullptr) {
      Timer timer;
      useCoverages = true;
      Logger::getInstance().addInfo("Using coverages.").print();

      if (maxIterations == std::numeric_limits<unsigned>::max() &&
          coverageDeltaThreshold == 0.) {
        maxIterations = 10;
        Logger::getInstance()
            .addWarning("No coverage initialization parameters set. Using " +
                        std::to_string(maxIterations) +
                        " initialization iterations.")
            .print();
      }

      if (!coveragesInitialized_) {
        timer.start();
        Logger::getInstance().addInfo("Initializing coverages ... ").print();
        // debug output
        if (logLevel >= 5)
          covMetricFile.open(name + "_covMetric.txt");

        for (unsigned iteration = 0; iteration < maxIterations; ++iteration) {
          // We need additional signal handling when running the C++ code from
          // the Python bindings to allow interrupts in the Python scripts
#ifdef VIENNAPS_PYTHON_BUILD
          if (PyErr_CheckSignals() != 0)
            throw pybind11::error_already_set();
#endif
          // save current coverages to compare with the new ones
          auto prevStepCoverages =
              SmartPointer<viennals::PointData<NumericType>>::New(*coverages);

          auto fluxes =
              calculateFluxes(rayTracer, useCoverages, useProcessParams);

          // update coverages in the model
          model->getSurfaceModel()->updateCoverages(fluxes, materialIds);

          // metric to check for convergence
          coverages = model->getSurfaceModel()->getCoverages();
          auto metric =
              calculateCoverageDeltaMetric(coverages, prevStepCoverages);

          if (logLevel >= 3) {
            mergeScalarData(mesh->getCellData(), coverages);
            mergeScalarData(mesh->getCellData(), fluxes);
            auto surfaceData = model->getSurfaceModel()->getSurfaceData();
            if (surfaceData)
              mergeScalarData(mesh->getCellData(), surfaceData);
            printDiskMesh(mesh, name + "_covInit_" + std::to_string(iteration) +
                                    ".vtp");

            Logger::getInstance()
                .addInfo("Iteration: " + std::to_string(iteration + 1))
                .print();

            std::stringstream stream;
            stream << std::setprecision(4) << std::fixed;
            stream << "Coverage delta metric: ";
            for (int i = 0; i < coverages->getScalarDataSize(); i++) {
              stream << coverages->getScalarDataLabel(i) << ": " << metric[i]
                     << "\t";
            }
            Logger::getInstance().addInfo(stream.str()).print();

            // log metric to file
            if (logLevel >= 5) {
              for (auto val : metric) {
                covMetricFile << val << ";";
              }
              covMetricFile << "\n";
            }
          }

          if (checkCoveragesConvergence(metric)) {
            Logger::getInstance()
                .addInfo("Coverages converged after " +
                         std::to_string(iteration + 1) + " iterations.")
                .print();
            break;
          }
        } // end coverage initialization loop
        coveragesInitialized_ = true;

        if (logLevel >= 5)
          covMetricFile.close();

        timer.finish();
        Logger::getInstance()
            .addTiming("Coverage initialization", timer)
            .print();
      }
    } // end coverage initialization

    auto fluxes = calculateFluxes(rayTracer, useCoverages, useProcessParams);
    mergeScalarData(mesh->getCellData(), fluxes);
    auto surfaceData = model->getSurfaceModel()->getSurfaceData();
    if (surfaceData)
      mergeScalarData(mesh->getCellData(), surfaceData);

    return mesh;
  }

  // Run the process.
  void apply() {
    /* ---------- Check input --------- */
    if (!domain) {
      Logger::getInstance().addWarning("No domain passed to Process.").print();
      return;
    }

    if (domain->getLevelSets().empty()) {
      Logger::getInstance().addWarning("No level sets in domain.").print();
      return;
    }

    if (!model) {
      Logger::getInstance()
          .addWarning("No process model passed to Process.")
          .print();
      return;
    }
    model->initialize(domain, processDuration);
    const auto name = model->getProcessName().value_or("default");

    if (model->getGeometricModel()) {
      model->getGeometricModel()->setDomain(domain);
      Logger::getInstance().addInfo("Applying geometric model...").print();
      model->getGeometricModel()->apply();
      return;
    }

    if (processDuration == 0.) {
      // apply only advection callback
      if (model->getAdvectionCallback()) {
        model->getAdvectionCallback()->setDomain(domain);
        model->getAdvectionCallback()->applyPreAdvect(0);
      } else {
        Logger::getInstance()
            .addWarning("No advection callback passed to Process.")
            .print();
      }
      return;
    }

    if (!model->getSurfaceModel()) {
      Logger::getInstance()
          .addWarning("No surface model passed to Process.")
          .print();
      return;
    }

    if (!model->getVelocityField()) {
      Logger::getInstance()
          .addWarning("No velocity field passed to Process.")
          .print();
      return;
    }

    /* ------ Process Setup ------ */
    const unsigned int logLevel = Logger::getLogLevel();
    Timer processTimer;
    processTimer.start();

    double remainingTime = processDuration;
    const NumericType gridDelta = domain->getGrid().getGridDelta();

    auto diskMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
    auto translator = SmartPointer<translatorType>::New();
    viennals::ToDiskMesh<NumericType, D> meshConverter(diskMesh);
    meshConverter.setTranslator(translator);
    if (domain->getMaterialMap() &&
        domain->getMaterialMap()->size() == domain->getLevelSets().size()) {
      meshConverter.setMaterialMap(domain->getMaterialMap()->getMaterialMap());
    }

    auto transField = SmartPointer<TranslationField<NumericType, D>>::New(
        model->getVelocityField(), domain->getMaterialMap());
    transField->setTranslator(translator);

    viennals::Advect<NumericType, D> advectionKernel;
    advectionKernel.setVelocityField(transField);
    advectionKernel.setIntegrationScheme(advectionParams.integrationScheme);
    advectionKernel.setTimeStepRatio(advectionParams.timeStepRatio);
    advectionKernel.setSaveAdvectionVelocities(advectionParams.velocityOutput);
    advectionKernel.setDissipationAlpha(advectionParams.dissipationAlpha);
    advectionKernel.setIgnoreVoids(advectionParams.ignoreVoids);
    advectionKernel.setCheckDissipation(advectionParams.checkDissipation);
    // normals vectors are only necessary for analytical velocity fields
    if (model->getVelocityField()->getTranslationFieldOptions() > 0)
      advectionKernel.setCalculateNormalVectors(false);

    for (auto dom : domain->getLevelSets()) {
      meshConverter.insertNextLevelSet(dom);
      advectionKernel.insertNextLevelSet(dom);
    }

    /* --------- Setup for ray tracing ----------- */
    const bool useRayTracing = !model->getParticleTypes().empty();

    viennaray::BoundaryCondition rayBoundaryCondition[D];
    viennaray::Trace<NumericType, D> rayTracer;

    if (useRayTracing) {
      initializeRayTracer(rayTracer);

      // initialize particle data logs
      particleDataLogs.resize(model->getParticleTypes().size());
      for (std::size_t i = 0; i < model->getParticleTypes().size(); i++) {
        int logSize = model->getParticleLogSize(i);
        if (logSize > 0) {
          particleDataLogs[i].data.resize(1);
          particleDataLogs[i].data[0].resize(logSize);
        }
      }
    }

    // Determine whether advection callback is used
    const bool useAdvectionCallback = model->getAdvectionCallback() != nullptr;
    if (useAdvectionCallback) {
      model->getAdvectionCallback()->setDomain(domain);
    }

    // Determine whether there are process parameters used in ray tracing
    model->getSurfaceModel()->initializeProcessParameters();
    const bool useProcessParams =
        model->getSurfaceModel()->getProcessParameters() != nullptr;

    if (useProcessParams)
      Logger::getInstance().addInfo("Using process parameters.").print();
    if (useAdvectionCallback)
      Logger::getInstance().addInfo("Using advection callback.").print();

    bool useCoverages = false;

    // Initialize coverages
    meshConverter.apply();
    auto numPoints = diskMesh->getNodes().size();
    model->getSurfaceModel()->initializeSurfaceData(numPoints);
    if (!coveragesInitialized_)
      model->getSurfaceModel()->initializeCoverages(numPoints);
    auto coverages = model->getSurfaceModel()->getCoverages();
    std::ofstream covMetricFile;
    if (coverages != nullptr) {
      Timer timer;
      useCoverages = true;
      Logger::getInstance().addInfo("Using coverages.").print();

      if (maxIterations == std::numeric_limits<unsigned>::max() &&
          coverageDeltaThreshold == 0.) {
        maxIterations = 10;
        Logger::getInstance()
            .addWarning("No coverage initialization parameters set. Using " +
                        std::to_string(maxIterations) +
                        " initialization iterations.")
            .print();
      }

      // debug output
      if (logLevel >= 5)
        covMetricFile.open(name + "_covMetric.txt");

      if (!coveragesInitialized_) {
        timer.start();
        Logger::getInstance().addInfo("Initializing coverages ... ").print();

        auto points = diskMesh->getNodes();
        auto normals = *diskMesh->getCellData().getVectorData("Normals");
        auto materialIds =
            *diskMesh->getCellData().getScalarData("MaterialIds");
        if (rayTracingParams.diskRadius == 0.) {
          rayTracer.setGeometry(points, normals, gridDelta);
        } else {
          rayTracer.setGeometry(points, normals, gridDelta,
                                rayTracingParams.diskRadius);
        }
        rayTracer.setMaterialIds(materialIds);

        model->initialize(domain, -1.);

        for (unsigned iteration = 0; iteration < maxIterations; ++iteration) {
          // We need additional signal handling when running the C++ code from
          // the Python bindings to allow interrupts in the Python scripts
#ifdef VIENNAPS_PYTHON_BUILD
          if (PyErr_CheckSignals() != 0)
            throw pybind11::error_already_set();
#endif
          // save current coverages to compare with the new ones
          auto prevStepCoverages =
              SmartPointer<viennals::PointData<NumericType>>::New(*coverages);

          auto fluxes =
              calculateFluxes(rayTracer, useCoverages, useProcessParams);

          // update coverages in the model
          model->getSurfaceModel()->updateCoverages(fluxes, materialIds);

          // metric to check for convergence
          coverages = model->getSurfaceModel()->getCoverages();
          auto metric =
              calculateCoverageDeltaMetric(coverages, prevStepCoverages);

          if (logLevel >= 3) {
            mergeScalarData(diskMesh->getCellData(), coverages);
            mergeScalarData(diskMesh->getCellData(), fluxes);
            auto surfaceData = model->getSurfaceModel()->getSurfaceData();
            if (surfaceData)
              mergeScalarData(diskMesh->getCellData(), surfaceData);
            printDiskMesh(diskMesh, name + "_covInit_" +
                                        std::to_string(iteration) + ".vtp");
            Logger::getInstance()
                .addInfo("Iteration: " + std::to_string(iteration + 1))
                .print();

            std::stringstream stream;
            stream << std::setprecision(4) << std::fixed;
            stream << "Coverage delta metric: ";
            for (int i = 0; i < coverages->getScalarDataSize(); i++) {
              stream << coverages->getScalarDataLabel(i) << ": " << metric[i]
                     << "\t";
            }
            Logger::getInstance().addInfo(stream.str()).print();

            // log metric to file
            if (logLevel >= 5) {
              for (auto val : metric) {
                covMetricFile << val << ";";
              }
              covMetricFile << "\n";
            }
          }

          if (checkCoveragesConvergence(metric)) {
            Logger::getInstance()
                .addInfo("Coverages converged after " +
                         std::to_string(iteration + 1) + " iterations.")
                .print();
            break;
          }
        } // end coverage initialization loop
        coveragesInitialized_ = true;

        timer.finish();
        Logger::getInstance()
            .addTiming("Coverage initialization", timer)
            .print();
      }
    } // end coverage initialization

    double previousTimeStep = 0.;
    size_t counter = 0;
    unsigned lsVelCounter = 0;
    Timer rtTimer;
    Timer callbackTimer;
    Timer advTimer;
    while (remainingTime > 0.) {
      // We need additional signal handling when running the C++ code from the
      // Python bindings to allow interrupts in the Python scripts
#ifdef VIENNAPS_PYTHON_BUILD
      if (PyErr_CheckSignals() != 0)
        throw pybind11::error_already_set();
#endif
      // Expand LS based on the integration scheme
      advectionKernel.prepareLS();
      model->initialize(domain, remainingTime);

      auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
      meshConverter.apply();
      auto materialIds = *diskMesh->getCellData().getScalarData("MaterialIds");
      auto points = diskMesh->getNodes();

      // rate calculation by top-down ray tracing
      if (useRayTracing) {
        rtTimer.start();
        auto normals = *diskMesh->getCellData().getVectorData("Normals");
        if (rayTracingParams.diskRadius == 0.) {
          rayTracer.setGeometry(points, normals, gridDelta);
        } else {
          rayTracer.setGeometry(points, normals, gridDelta,
                                rayTracingParams.diskRadius);
        }
        rayTracer.setMaterialIds(materialIds);

        fluxes = calculateFluxes(rayTracer, useCoverages, useProcessParams);

        rtTimer.finish();
        Logger::getInstance()
            .addTiming("Top-down flux calculation", rtTimer)
            .print();
      }

      // update coverages and calculate coverage delta metric
      if (useCoverages) {
        coverages = model->getSurfaceModel()->getCoverages();
        auto prevStepCoverages =
            SmartPointer<viennals::PointData<NumericType>>::New(*coverages);

        // update coverages in the model
        model->getSurfaceModel()->updateCoverages(fluxes, materialIds);

        if (coverageDeltaThreshold > 0) {
          auto metric =
              calculateCoverageDeltaMetric(coverages, prevStepCoverages);
          while (!checkCoveragesConvergence(metric)) {
            Logger::getInstance()
                .addInfo("Coverages did not converge. Repeating flux "
                         "calculation.")
                .print();

            prevStepCoverages =
                SmartPointer<viennals::PointData<NumericType>>::New(*coverages);

            rtTimer.start();
            fluxes = calculateFluxes(rayTracer, useCoverages, useProcessParams);
            rtTimer.finish();
            model->getSurfaceModel()->updateCoverages(fluxes, materialIds);

            coverages = model->getSurfaceModel()->getCoverages();
            metric = calculateCoverageDeltaMetric(coverages, prevStepCoverages);
            if (logLevel >= 5) {
              for (auto val : metric) {
                covMetricFile << val << ";";
              }
              covMetricFile << "\n";
            }

            Logger::getInstance()
                .addTiming("Top-down flux calculation", rtTimer)
                .print();
          }
        }
      }

      // calculate velocities from fluxes
      auto velocities = model->getSurfaceModel()->calculateVelocities(
          fluxes, points, materialIds);

      // prepare velocity field
      model->getVelocityField()->prepare(domain, velocities,
                                         processDuration - remainingTime);
      if (model->getVelocityField()->getTranslationFieldOptions() == 2)
        transField->buildKdTree(points);

      // print debug output
      if (logLevel >= 4) {
        if (velocities)
          diskMesh->getCellData().insertNextScalarData(*velocities,
                                                       "velocities");
        if (useCoverages) {
          auto coverages = model->getSurfaceModel()->getCoverages();
          mergeScalarData(diskMesh->getCellData(), coverages);
        }
        auto surfaceData = model->getSurfaceModel()->getSurfaceData();
        if (surfaceData)
          mergeScalarData(diskMesh->getCellData(), surfaceData);
        mergeScalarData(diskMesh->getCellData(), fluxes);
        printDiskMesh(diskMesh, name + "_" + std::to_string(counter) + ".vtp");
        if (domain->getCellSet()) {
          domain->getCellSet()->writeVTU(name + "_cellSet_" +
                                         std::to_string(counter) + ".vtu");
        }
        counter++;
      }

      // apply advection callback
      if (useAdvectionCallback) {
        callbackTimer.start();
        bool continueProcess = model->getAdvectionCallback()->applyPreAdvect(
            processDuration - remainingTime);
        callbackTimer.finish();
        Logger::getInstance()
            .addTiming("Advection callback pre-advect", callbackTimer)
            .print();

        if (!continueProcess) {
          Logger::getInstance()
              .addInfo("Process stopped early by AdvectionCallback during "
                       "`preAdvect`.")
              .print();
          break;
        }
      }

      // adjust time step near end
      if (remainingTime - previousTimeStep < 0.) {
        advectionKernel.setAdvectionTime(remainingTime);
      }

      // move coverages to LS, so they are moved with the advection step
      if (useCoverages)
        moveCoveragesToTopLS(translator,
                             model->getSurfaceModel()->getCoverages());
      advTimer.start();
      advectionKernel.apply();
      advTimer.finish();
      Logger::getInstance().addTiming("Surface advection", advTimer).print();

      if (advectionParams.velocityOutput) {
        auto lsMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
        viennals::ToMesh<NumericType, D>(domain->getLevelSets().back(), lsMesh)
            .apply();
        viennals::VTKWriter<NumericType>(
            lsMesh, "ls_velocities_" + std::to_string(lsVelCounter++) + ".vtp")
            .apply();
      }

      // update the translator to retrieve the correct coverages from the LS
      meshConverter.apply();
      if (useCoverages)
        updateCoveragesFromAdvectedSurface(
            translator, model->getSurfaceModel()->getCoverages());

      // apply advection callback
      if (useAdvectionCallback) {
        callbackTimer.start();
        bool continueProcess = model->getAdvectionCallback()->applyPostAdvect(
            advectionKernel.getAdvectedTime());
        callbackTimer.finish();
        Logger::getInstance()
            .addTiming("Advection callback post-advect", callbackTimer)
            .print();
        if (!continueProcess) {
          Logger::getInstance()
              .addInfo("Process stopped early by AdvectionCallback during "
                       "`postAdvect`.")
              .print();
          break;
        }
      }

      previousTimeStep = advectionKernel.getAdvectedTime();
      if (previousTimeStep == std::numeric_limits<NumericType>::max()) {
        Logger::getInstance()
            .addInfo("Process halted: Surface velocities are zero across the "
                     "entire surface.")
            .print();
        remainingTime = 0.;
        break;
      }
      remainingTime -= previousTimeStep;

      if (Logger::getLogLevel() >= 2) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(4)
               << "Process time: " << processDuration - remainingTime << " / "
               << processDuration << " "
               << units::Time::getInstance().toShortString();
        Logger::getInstance().addInfo(stream.str()).print();
      }
    }

    processTime = processDuration - remainingTime;
    processTimer.finish();

    Logger::getInstance()
        .addTiming("\nProcess " + name, processTimer)
        .addTiming("Surface advection total time",
                   advTimer.totalDuration * 1e-9,
                   processTimer.totalDuration * 1e-9)
        .print();
    if (useRayTracing) {
      Logger::getInstance()
          .addTiming("Top-down flux calculation total time",
                     rtTimer.totalDuration * 1e-9,
                     processTimer.totalDuration * 1e-9)
          .print();
    }
    if (useAdvectionCallback) {
      Logger::getInstance()
          .addTiming("Advection callback total time",
                     callbackTimer.totalDuration * 1e-9,
                     processTimer.totalDuration * 1e-9)
          .print();
    }
    model->reset();
    if (logLevel >= 5)
      covMetricFile.close();
  }

  void writeParticleDataLogs(std::string fileName) {
    std::ofstream file(fileName.c_str());

    for (std::size_t i = 0; i < particleDataLogs.size(); i++) {
      if (!particleDataLogs[i].data.empty()) {
        file << "particle" << i << "_data ";
        for (std::size_t j = 0; j < particleDataLogs[i].data[0].size(); j++) {
          file << particleDataLogs[i].data[0][j] << " ";
        }
        file << "\n";
      }
    }

    file.close();
  }

private:
  static void printDiskMesh(SmartPointer<viennals::Mesh<NumericType>> mesh,
                            std::string name) {
    viennals::VTKWriter<NumericType>(mesh, std::move(name)).apply();
  }

  static void
  mergeScalarData(viennals::PointData<NumericType> &scalarData,
                  SmartPointer<viennals::PointData<NumericType>> dataToInsert) {
    int numScalarData = dataToInsert->getScalarDataSize();
    for (int i = 0; i < numScalarData; i++) {
      scalarData.insertReplaceScalarData(*dataToInsert->getScalarData(i),
                                         dataToInsert->getScalarDataLabel(i));
    }
  }

  static viennaray::TracingData<NumericType> movePointDataToRayData(
      SmartPointer<viennals::PointData<NumericType>> pointData) {
    viennaray::TracingData<NumericType> rayData;
    const auto numData = pointData->getScalarDataSize();
    rayData.setNumberOfVectorData(numData);
    for (size_t i = 0; i < numData; ++i) {
      auto label = pointData->getScalarDataLabel(i);
      rayData.setVectorData(i, std::move(*pointData->getScalarData(label)),
                            label);
    }

    return std::move(rayData);
  }

  static void moveRayDataToPointData(
      SmartPointer<viennals::PointData<NumericType>> pointData,
      viennaray::TracingData<NumericType> &rayData) {
    pointData->clear();
    const auto numData = rayData.getVectorData().size();
    for (size_t i = 0; i < numData; ++i)
      pointData->insertNextScalarData(std::move(rayData.getVectorData(i)),
                                      rayData.getVectorDataLabel(i));
  }

  void moveCoveragesToTopLS(
      SmartPointer<translatorType> translator,
      SmartPointer<viennals::PointData<NumericType>> coverages) {
    auto topLS = domain->getLevelSets().back();
    for (size_t i = 0; i < coverages->getScalarDataSize(); i++) {
      auto covName = coverages->getScalarDataLabel(i);
      std::vector<NumericType> levelSetData(topLS->getNumberOfPoints(), 0);
      auto cov = coverages->getScalarData(covName);
      for (const auto iter : *translator.get()) {
        levelSetData[iter.first] = cov->at(iter.second);
      }
      if (auto data = topLS->getPointData().getScalarData(covName, true);
          data != nullptr) {
        *data = std::move(levelSetData);
      } else {
        topLS->getPointData().insertNextScalarData(std::move(levelSetData),
                                                   covName);
      }
    }
  }

  void updateCoveragesFromAdvectedSurface(
      SmartPointer<translatorType> translator,
      SmartPointer<viennals::PointData<NumericType>> coverages) const {
    auto topLS = domain->getLevelSets().back();
    for (size_t i = 0; i < coverages->getScalarDataSize(); i++) {
      auto covName = coverages->getScalarDataLabel(i);
      auto levelSetData = topLS->getPointData().getScalarData(covName);
      auto covData = coverages->getScalarData(covName);
      covData->resize(translator->size());
      for (const auto it : *translator.get()) {
        covData->at(it.second) = levelSetData->at(it.first);
      }
    }
  }

  static std::vector<NumericType> calculateCoverageDeltaMetric(
      SmartPointer<viennals::PointData<NumericType>> updated,
      SmartPointer<viennals::PointData<NumericType>> previous) {

    assert(updated->getScalarDataSize() == previous->getScalarDataSize());
    std::vector<NumericType> delta(updated->getScalarDataSize(), 0.);

#pragma omp parallel for
    for (int i = 0; i < updated->getScalarDataSize(); i++) {
      auto label = updated->getScalarDataLabel(i);
      auto updatedData = updated->getScalarData(label);
      auto previousData = previous->getScalarData(label);
      for (size_t j = 0; j < updatedData->size(); j++) {
        auto diff = updatedData->at(j) - previousData->at(j);
        delta[i] += diff * diff;
      }

      delta[i] /= updatedData->size();
    }

    return delta;
  }

  bool
  checkCoveragesConvergence(const std::vector<NumericType> &deltaMetric) const {
    for (auto val : deltaMetric) {
      if (val > coverageDeltaThreshold)
        return false;
    }
    return true;
  }

  void initializeRayTracer(viennaray::Trace<NumericType, D> &tracer) const {
    // Map the domain boundary to the ray tracing boundaries
    viennaray::BoundaryCondition rayBoundaryCondition[D];
    if (rayTracingParams.ignoreFluxBoundaries) {
      for (unsigned i = 0; i < D; ++i)
        rayBoundaryCondition[i] = viennaray::BoundaryCondition::IGNORE;
    } else {
      for (unsigned i = 0; i < D; ++i)
        rayBoundaryCondition[i] = utils::convertBoundaryCondition(
            domain->getGrid().getBoundaryConditions(i));
    }
    tracer.setBoundaryConditions(rayBoundaryCondition);
    tracer.setSourceDirection(rayTracingParams.sourceDirection);
    tracer.setNumberOfRaysPerPoint(rayTracingParams.raysPerPoint);
    tracer.setUseRandomSeeds(rayTracingParams.useRandomSeeds);
    tracer.setCalculateFlux(false);

    auto source = model->getSource();
    if (source) {
      tracer.setSource(source);
      Logger::getInstance().addInfo("Using custom source.").print();
    }
    auto primaryDirection = model->getPrimaryDirection();
    if (primaryDirection) {
      Logger::getInstance()
          .addInfo("Using primary direction: " +
                   utils::arrayToString(primaryDirection.value()))
          .print();
      tracer.setPrimaryDirection(primaryDirection.value());
    }
  }

  auto calculateFluxes(viennaray::Trace<NumericType, D> &rayTracer,
                       const bool useCoverages, const bool useProcessParams) {

    viennaray::TracingData<NumericType> rayTracingData;

    // move coverages to the ray tracer
    if (useCoverages) {
      rayTracingData =
          movePointDataToRayData(model->getSurfaceModel()->getCoverages());
    }

    if (useProcessParams) {
      // store scalars in addition to coverages
      auto processParams = model->getSurfaceModel()->getProcessParameters();
      NumericType numParams = processParams->getScalarData().size();
      rayTracingData.setNumberOfScalarData(numParams);
      for (size_t i = 0; i < numParams; ++i) {
        rayTracingData.setScalarData(i, processParams->getScalarData(i),
                                     processParams->getScalarDataLabel(i));
      }
    }

    if (useCoverages || useProcessParams)
      rayTracer.setGlobalData(rayTracingData);

    auto fluxes = runRayTracer(rayTracer);

    // move coverages back in the model
    if (useCoverages)
      moveRayDataToPointData(model->getSurfaceModel()->getCoverages(),
                             rayTracingData);

    return fluxes;
  }

  auto runRayTracer(viennaray::Trace<NumericType, D> &rayTracer) {
    auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
    unsigned particleIdx = 0;
    for (auto &particle : model->getParticleTypes()) {
      int dataLogSize = model->getParticleLogSize(particleIdx);
      if (dataLogSize > 0) {
        rayTracer.getDataLog().data.resize(1);
        rayTracer.getDataLog().data[0].resize(dataLogSize, 0.);
      }
      rayTracer.setParticleType(particle);
      rayTracer.apply();

      // fill up fluxes vector with fluxes from this particle type
      auto &localData = rayTracer.getLocalData();
      int numFluxes = particle->getLocalDataLabels().size();
      for (int i = 0; i < numFluxes; ++i) {
        auto flux = std::move(localData.getVectorData(i));

        // normalize and smooth
        rayTracer.normalizeFlux(flux, rayTracingParams.normalizationType);
        if (rayTracingParams.smoothingNeighbors > 0)
          rayTracer.smoothFlux(flux, rayTracingParams.smoothingNeighbors);

        fluxes->insertNextScalarData(std::move(flux),
                                     localData.getVectorDataLabel(i));
      }

      if (dataLogSize > 0) {
        particleDataLogs[particleIdx].merge(rayTracer.getDataLog());
      }
      ++particleIdx;
    }
    return fluxes;
  }

private:
  psDomainType domain;
  SmartPointer<ProcessModel<NumericType, D>> model;
  std::vector<viennaray::DataLog<NumericType>> particleDataLogs;
  NumericType processDuration;
  unsigned maxIterations = std::numeric_limits<unsigned>::max();
  NumericType coverageDeltaThreshold = 0.;
  bool coveragesInitialized_ = false;
  NumericType processTime = 0.;

  AdvectionParameters<NumericType> advectionParams;
  RayTracingParameters<NumericType, D> rayTracingParams;
};

} // namespace viennaps
