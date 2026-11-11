#pragma once

#include "psProcessContext.hpp"

#include <lsAdvect.hpp>
#include <lsAdvectHermite.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsMaterialStack.hpp>

#include <vcTimer.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace viennaps {

VIENNAPS_TEMPLATE_ND(NumericType, D) class AdvectionHandler {
  // Advect, on a Euclidean surface with values only for the upwind update
  // (setEuclideanRenormalization)
  viennals::Advect<NumericType, D> advectionKernel_;
  // a Euclidean surface keeps its Hermite data through AdvectHermite with the
  // Hermite update, one time step per call as Advect's single step
  viennals::AdvectHermite<NumericType, D> hermiteKernel_;
  bool hermite_ = false;
  // the distance after which the upwind update takes over, and the distance
  // the fastest point of the surface has moved in the process
  double upwindAfter_ = 0., frontDistance_ = 0.;
  viennacore::Timer<> timer_;
  unsigned lsVelOutputCounter_ = 0;
  unsigned totalAdvectionSteps_ = 0;

public:
  ProcessResult initialize(ProcessContext<NumericType, D> &context) {
    // Initialize advection handler with context
    assert(context.translationField);
    auto translationMethod = context.translationField->getTranslationMethod();
    if (translationMethod > 2 || translationMethod < 0) {
      VIENNACORE_LOG_WARNING("Translation field method not supported.");
      return ProcessResult::INVALID_INPUT;
    }

    auto &discSchem = context.advectionParams.spatialScheme;
    if (translationMethod == 1 &&
        (discSchem != SpatialScheme::ENGQUIST_OSHER_1ST_ORDER &&
         discSchem != SpatialScheme::ENGQUIST_OSHER_2ND_ORDER &&
         discSchem != SpatialScheme::LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER &&
         discSchem != SpatialScheme::LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER &&
         discSchem != SpatialScheme::WENO_3RD_ORDER &&
         discSchem != SpatialScheme::WENO_5TH_ORDER)) {
      VIENNACORE_LOG_WARNING(
          "Translation field method not supported in combination "
          "with discretization scheme.");
      return ProcessResult::INVALID_INPUT;
    }

    context.resetTime();

    advectionKernel_.setSingleStep(true);
    advectionKernel_.setSpatialScheme(context.advectionParams.spatialScheme);
    advectionKernel_.setTemporalScheme(context.advectionParams.temporalScheme);
    advectionKernel_.setVelocityField(context.translationField);
    advectionKernel_.setTimeStepRatio(context.advectionParams.timeStepRatio);
    advectionKernel_.setSaveAdvectionVelocities(
        context.advectionParams.velocityOutput);
    advectionKernel_.setDissipationAlpha(
        context.advectionParams.dissipationAlpha);
    advectionKernel_.setIgnoreVoids(context.advectionParams.ignoreVoids);
    advectionKernel_.setCheckDissipation(
        context.advectionParams.checkDissipation);
    advectionKernel_.setAdaptiveTimeStepping(
        context.advectionParams.adaptiveTimeStepping,
        context.advectionParams.adaptiveTimeStepSubdivisions);

    advectionKernel_.setVelocityUpdateCallback(nullptr);
    advectionKernel_.setEuclideanRenormalization(true);

    // normals vectors are only necessary for analytical velocity fields
    if (translationMethod > 0)
      advectionKernel_.setCalculateNormalVectors(false);

    advectionKernel_.clearLevelSets();
    for (auto &dom : context.domain->getLevelSets()) {
      advectionKernel_.insertNextLevelSet(dom);
    }

    hermite_ = context.isEuclidean();
    if (hermite_) {
      hermiteKernel_.setSingleStep(true);
      hermiteKernel_.setVelocityField(context.translationField);
      hermiteKernel_.setTimeStepRatio(context.advectionParams.timeStepRatio);
      // ray-traced fluxes give speeds only for the surface's own normals
      hermiteKernel_.setSurfaceSpeedsOnly(context.flags.useFluxEngine);
      // the upwind update where the model asks for it (plasma etching, TEOS
      // deposition), from the start (the Process drops the Hermite data
      // first) or once the front has moved the model's distance
      // (performAdvection): Advect with values only and the advection
      // parameters' schemes
      upwindAfter_ = context.model->getEuclideanUpwindAfter() *
                     context.domain->getGridDelta();
      frontDistance_ = 0.;
      context.euclideanUpwind =
          context.model->getEuclideanUpwind() && upwindAfter_ <= 0.;
      hermiteKernel_.clearLevelSets();
      for (auto &dom : context.domain->getLevelSets())
        hermiteKernel_.insertNextLevelSet(dom);
    }

    totalAdvectionSteps_ = 0;

    return ProcessResult::SUCCESS;
  }

  void setAdvectionTime(double time) {
    advectionKernel_.setAdvectionTime(time);
    hermiteKernel_.setAdvectionTime(time);
  }

  void setVelocityUpdateCallback(
      std::function<bool(SmartPointer<viennals::Domain<NumericType, D>>)>
          callback) {
    advectionKernel_.setVelocityUpdateCallback(callback);
  }

  auto getTotalAdvectionSteps() const { return totalAdvectionSteps_; }

  void disableSingleStep() {
    advectionKernel_.setSingleStep(false);
    hermiteKernel_.setSingleStep(false);
  }

  void prepareAdvection(const ProcessContext<NumericType, D> &context) {
    // Prepare for advection step; the band of AdvectHermite already holds
    // the width it needs
    if (!context.isHermite())
      advectionKernel_.prepareLS();
    context.model->initialize(context.domain, context.processTime);
  }

  ProcessResult performAdvection(ProcessContext<NumericType, D> &context) {
    // Perform the advection step

    // Set the maximum advection time.
    if (!context.flags.isALP) {
      advectionKernel_.setAdvectionTime(context.processDuration -
                                        context.processTime);
      hermiteKernel_.setAdvectionTime(context.processDuration -
                                      context.processTime);
    }

    timer_.start();
    const bool hermiteStep = context.isHermite();
    if (hermiteStep)
      hermiteKernel_.apply();
    else
      advectionKernel_.apply();
    timer_.finish();

    ++totalAdvectionSteps_;
    if (context.advectionParams.velocityOutput) {
      auto mesh = viennals::Mesh<NumericType>::New();
      viennals::ToMesh<NumericType, D>(context.domain->getSurface(), mesh)
          .apply();
      viennals::VTKWriter<NumericType>(
          mesh,
          "ls_velocities_" + std::to_string(lsVelOutputCounter_++) + ".vtp")
          .apply();
    }

    context.timeStep = hermiteStep ? hermiteKernel_.getAdvectedTime()
                                   : advectionKernel_.getAdvectedTime();
    if (context.timeStep == std::numeric_limits<double>::max()) {
      VIENNACORE_LOG_WARNING(
          "Process terminated early: Velocities are zero everywhere.");
      context.processTime = context.processDuration;
    } else {
      context.processTime += context.timeStep;
      // the upwind update takes over once the fastest point of the surface
      // has moved the model's distance
      if (hermite_ && !context.euclideanUpwind &&
          context.model->getEuclideanUpwind()) {
        frontDistance_ += context.stepMaxSpeed * context.timeStep;
        context.euclideanUpwind = frontDistance_ >= upwindAfter_;
        if (context.euclideanUpwind)
          for (auto &levelSet : context.domain->getLevelSets())
            viennals::dropHermiteData(levelSet);
      }
    }

    return ProcessResult::SUCCESS;
  }

  ProcessResult copyCoveragesToLevelSet(
      const ProcessContext<NumericType, D> &context,
      SmartPointer<std::unordered_map<unsigned long, unsigned long>> const
          &translator) {
    // Move coverages to the top level set
    auto topLS = context.domain->getSurface();
    auto coverages = context.model->getSurfaceModel()->getCoverages();
    assert(coverages != nullptr);
    assert(translator != nullptr);

    std::vector<std::vector<NumericType>> levelSetCoverages(
        coverages->getScalarDataSize());

    // a Euclidean band moves nodes that are not on the surface now, so every
    // node of it takes the coverages of a disk, as the velocities do: with
    // AdvectHermite the disk nearest to the node's surface point among the
    // disks of the material behind that point (Hermite data); with Advect's
    // values only, the disk nearest to the node
    std::vector<long> nearestDisk;
    if (context.isEuclidean() && context.translationField->getKdTree()) {
      const auto gridDelta = topLS->getGrid().getGridDelta();
      const auto *normals =
          context.isHermite()
              ? topLS->getPointData().getVectorData(
                    viennals::FromSurfaceMesh<NumericType,
                                              D>::surfaceNormalsLabel,
                    true)
              : nullptr;
      const auto &levelSets = context.domain->getLevelSets();
      std::unique_ptr<const viennals::MaterialStack<NumericType, D>> stack;
      if (normals != nullptr)
        stack = std::make_unique<const viennals::MaterialStack<NumericType, D>>(
            std::vector<SmartPointer<viennals::Domain<NumericType, D>>>(
                levelSets.begin(), levelSets.end() - 1));
      const auto &materialMap = context.domain->getMaterialMap();
      nearestDisk.assign(topLS->getNumberOfPoints(), -1);
      // the defined points of the band first, so that their queries run in
      // parallel
      std::vector<unsigned long> ids;
      std::vector<viennahrle::Index<D>> indices;
      std::vector<NumericType> values;
      ids.reserve(topLS->getNumberOfPoints());
      indices.reserve(topLS->getNumberOfPoints());
      values.reserve(topLS->getNumberOfPoints());
      for (viennahrle::ConstSparseIterator<
               typename viennals::Domain<NumericType, D>::DomainType>
               it(topLS->getDomain());
           !it.isFinished(); it.next()) {
        if (!it.isDefined())
          continue;
        ids.push_back(it.getPointId());
        indices.push_back(it.getStartIndices());
        values.push_back(it.getValue());
      }
#pragma omp parallel for
      for (long q = 0; q < long(ids.size()); ++q) {
        const auto id = ids[q];
        Vec3D<NumericType> point{}, behind{};
        for (int i = 0; i < D; ++i)
          point[i] = indices[q][i] * gridDelta;
        int material = -1;
        if (normals != nullptr) {
          const auto &n = (*normals)[id];
          const NumericType length =
              std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
          if (length > NumericType(1e-12)) {
            for (int i = 0; i < D; ++i) {
              point[i] -= values[q] * gridDelta * n[i] / length;
              behind[i] =
                  point[i] / gridDelta - NumericType(0.01) * n[i] / length;
            }
            const int index = stack->material(behind);
            material = materialMap ? int(materialMap->getMaterialIdAtIdx(index))
                                   : index;
          }
        }
        nearestDisk[id] =
            context.translationField->findNearestDisk(point, material);
      }
    }

#pragma omp parallel for
    for (unsigned i = 0; i < levelSetCoverages.size(); i++) {
      auto covName = coverages->getScalarDataLabel(i);
      std::vector<NumericType> levelSetData(topLS->getNumberOfPoints(), 0);
      auto cov = coverages->getScalarData(covName);

      if (!nearestDisk.empty()) {
        for (std::size_t lsId = 0; lsId < nearestDisk.size(); ++lsId)
          if (nearestDisk[lsId] >= 0)
            levelSetData[lsId] = cov->at(nearestDisk[lsId]);
      } else {
        for (const auto &[lsId, surfaceId] : *translator) {
          levelSetData[lsId] = cov->at(surfaceId);
        }
      }

      levelSetCoverages[i] = std::move(levelSetData);
    }

    for (unsigned i = 0; i < levelSetCoverages.size(); i++) {
      auto covName = coverages->getScalarDataLabel(i);
      topLS->getPointData().insertReplaceScalarData(
          std::move(levelSetCoverages[i]), covName);
    }

    return ProcessResult::SUCCESS;
  }

  ProcessResult updateCoveragesFromAdvectedSurface(
      const ProcessContext<NumericType, D> &context,
      SmartPointer<std::unordered_map<unsigned long, unsigned long>> const
          &translator) {
    // Update coverages from the advected surface
    auto topLS = context.domain->getSurface();
    auto coverages = context.model->getSurfaceModel()->getCoverages();
    assert(coverages != nullptr);
    assert(translator != nullptr);

    for (size_t i = 0; i < coverages->getScalarDataSize(); i++) {
      auto covName = coverages->getScalarDataLabel(i);
      auto levelSetData = topLS->getPointData().getScalarData(covName);
      auto covData = coverages->getScalarData(covName);
      covData->resize(translator->size());

      for (const auto &[lsId, surfaceId] : *translator) {
        covData->at(surfaceId) = levelSetData->at(lsId);
      }
    }

    return ProcessResult::SUCCESS;
  }
  auto &getTimer() const { return timer_; }
  void resetTimer() {
    timer_.reset();
    advectionKernel_.resetPhaseTimes();
    hermiteKernel_.resetPhaseTimes();
  }
  // the phases of the advection kernels that took time, named after the
  // kernel (Advect or AdvectHermite) and the phase
  std::vector<std::pair<std::string, double>> getPhaseTimes() const {
    std::vector<std::pair<std::string, double>> times;
    for (const auto &phase : advectionKernel_.getPhaseTimes())
      if (phase.second > 0.)
        times.emplace_back("Advect " + phase.first, phase.second);
    for (const auto &phase : hermiteKernel_.getPhaseTimes())
      if (phase.second > 0.)
        times.emplace_back("AdvectHermite " + phase.first, phase.second);
    return times;
  }
};

} // namespace viennaps
