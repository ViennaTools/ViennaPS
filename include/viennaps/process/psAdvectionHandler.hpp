#pragma once

#include "psProcessContext.hpp"

#include <lsAdvect.hpp>
#include <lsAdvectHermite.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsMaterialStack.hpp>

#include <vcTimer.hpp>

namespace viennaps {

VIENNAPS_TEMPLATE_ND(NumericType, D) class AdvectionHandler {
  viennals::Advect<NumericType, D> advectionKernel_;
  // a Euclidean surface keeps its Hermite data through AdvectHermite, one
  // time step per call as Advect's single step
  viennals::AdvectHermite<NumericType, D> hermiteKernel_;
  bool hermite_ = false;
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
    // Prepare for advection step; the Euclidean band already holds the
    // width it needs
    if (!hermite_)
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
    if (hermite_)
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

    context.timeStep = hermite_ ? hermiteKernel_.getAdvectedTime()
                                : advectionKernel_.getAdvectedTime();
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
    auto topLS = context.domain->getSurface();
    auto coverages = context.model->getSurfaceModel()->getCoverages();
    assert(coverages != nullptr);
    assert(translator != nullptr);

    std::vector<std::vector<NumericType>> levelSetCoverages(
        coverages->getScalarDataSize());

    // AdvectHermite moves the surface across nodes that are not on it now,
    // so every node of the Euclidean band takes the coverages of the disk
    // nearest to its surface point among the disks of the material behind
    // that point, as the velocities do
    std::vector<long> nearestDisk;
    if (hermite_ && context.translationField->getKdTree()) {
      const auto gridDelta = topLS->getGrid().getGridDelta();
      const auto *normals = topLS->getPointData().getVectorData(
          viennals::FromSurfaceMesh<NumericType, D>::surfaceNormalsLabel, true);
      const auto &levelSets = context.domain->getLevelSets();
      const viennals::MaterialStack<NumericType, D> stack(
          std::vector<SmartPointer<viennals::Domain<NumericType, D>>>(
              levelSets.begin(), levelSets.end() - 1));
      const auto &materialMap = context.domain->getMaterialMap();
      nearestDisk.assign(topLS->getNumberOfPoints(), -1);
      for (viennahrle::ConstSparseIterator<
               typename viennals::Domain<NumericType, D>::DomainType>
               it(topLS->getDomain());
           !it.isFinished(); it.next()) {
        if (!it.isDefined())
          continue;
        const auto id = it.getPointId();
        Vec3D<NumericType> point{}, behind{};
        for (int i = 0; i < D; ++i)
          point[i] = it.getStartIndices(i) * gridDelta;
        int material = -1;
        if (normals != nullptr) {
          const auto &n = (*normals)[id];
          const NumericType length =
              std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
          if (length > NumericType(1e-12)) {
            for (int i = 0; i < D; ++i) {
              point[i] -= it.getValue() * gridDelta * n[i] / length;
              behind[i] =
                  point[i] / gridDelta - NumericType(0.01) * n[i] / length;
            }
            const int index = stack.material(behind);
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
  void resetTimer() { timer_.reset(); }
};

} // namespace viennaps
