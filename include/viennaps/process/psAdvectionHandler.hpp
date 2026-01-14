#pragma once

#include "psProcessContext.hpp"

#include <lsAdvect.hpp>

#include <vcTimer.hpp>

namespace viennaps {

template <typename NumericType, int D> class AdvectionHandler {
  viennals::Advect<NumericType, D> advectionKernel_;
  viennacore::Timer<> timer_;
  unsigned lsVelOutputCounter = 0;

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
         discSchem != SpatialScheme::WENO_5TH_ORDER)) {
      VIENNACORE_LOG_WARNING(
          "Translation field method not supported in combination "
          "with discretization scheme.");
      return ProcessResult::INVALID_INPUT;
    }

    context.resetTime();

    advectionKernel_.setTemporalScheme(context.advectionParams.temporalScheme);

    advectionKernel_.setSingleStep(true);
    advectionKernel_.setVelocityField(context.translationField);
    advectionKernel_.setSpatialScheme(context.advectionParams.spatialScheme);
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

    // normals vectors are only necessary for analytical velocity fields
    if (translationMethod > 0)
      advectionKernel_.setCalculateNormalVectors(false);

    advectionKernel_.clearLevelSets();
    for (auto &dom : context.domain->getLevelSets()) {
      advectionKernel_.insertNextLevelSet(dom);
    }

    return ProcessResult::SUCCESS;
  }

  void setAdvectionTime(double time) {
    advectionKernel_.setAdvectionTime(time);
  }

  void disableSingleStep() { advectionKernel_.setSingleStep(false); }

  void prepareAdvection(const ProcessContext<NumericType, D> &context) {
    // Prepare for advection step
    advectionKernel_.prepareLS();
    context.model->initialize(context.domain, context.processTime);
  }

  ProcessResult performAdvection(ProcessContext<NumericType, D> &context) {
    // Perform the advection step

    // Set the maximum advection time.
    if (!context.flags.isALP) {
      advectionKernel_.setAdvectionTime(context.processDuration -
                                        context.processTime);
    }

    timer_.start();
    advectionKernel_.apply();
    timer_.finish();

    if (context.advectionParams.velocityOutput) {
      auto mesh = viennals::Mesh<NumericType>::New();
      viennals::ToMesh<NumericType, D>(context.domain->getLevelSets().back(),
                                       mesh)
          .apply();
      viennals::VTKWriter<NumericType>(
          mesh,
          "ls_velocities_" + std::to_string(lsVelOutputCounter++) + ".vtp")
          .apply();
    }

    context.timeStep = advectionKernel_.getAdvectedTime();
    if (context.timeStep == std::numeric_limits<double>::max()) {
      VIENNACORE_LOG_WARNING(
          "Process terminated early: Velocities are zero everywhere.");
      context.processTime = context.processDuration;
    } else {
      context.processTime += context.timeStep;
    }

    return ProcessResult::SUCCESS;
  }

  ProcessResult copyCoveragesToLevelSet(
      const ProcessContext<NumericType, D> &context,
      SmartPointer<std::unordered_map<unsigned long, unsigned long>> const
          &translator) {
    // Move coverages to the top level set
    auto topLS = context.domain->getLevelSets().back();
    auto coverages = context.model->getSurfaceModel()->getCoverages();
    assert(coverages != nullptr);
    assert(translator != nullptr);

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

    return ProcessResult::SUCCESS;
  }

  ProcessResult updateCoveragesFromAdvectedSurface(
      const ProcessContext<NumericType, D> &context,
      SmartPointer<std::unordered_map<unsigned long, unsigned long>> const
          &translator) {
    // Update coverages from the advected surface
    auto topLS = context.domain->getLevelSets().back();
    auto coverages = context.model->getSurfaceModel()->getCoverages();
    for (size_t i = 0; i < coverages->getScalarDataSize(); i++) {
      auto covName = coverages->getScalarDataLabel(i);
      auto levelSetData = topLS->getPointData().getScalarData(covName);
      auto covData = coverages->getScalarData(covName);
      covData->resize(translator->size());
      for (const auto it : *translator.get()) {
        covData->at(it.second) = levelSetData->at(it.first);
      }
    }
    return ProcessResult::SUCCESS;
  }
  auto &getTimer() const { return timer_; }
  void resetTimer() { timer_.reset(); }
};

} // namespace viennaps
