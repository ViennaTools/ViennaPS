#pragma once

#include "psProcessContext.hpp"
#include "psTranslationField.hpp"

#include <lsAdvect.hpp>

#include <vcTimer.hpp>

namespace viennaps {

template <typename NumericType, int D> class AdvectionHandler {
  viennals::Advect<NumericType, D> advectionKernel_;
  SmartPointer<TranslationField<NumericType, D>> translationField_ = nullptr;
  viennacore::Timer<> timer_;
  unsigned long lsVelCounter = 0;

public:
  ProcessResult initialize(const ProcessContext<NumericType, D> &context) {
    // Initialize advection handler with context
    translationField_ = SmartPointer<TranslationField<NumericType, D>>::New(
        context.model->getVelocityField(), context.domain->getMaterialMap());

    auto translationMethod = translationField_->getTranslationMethod();
    if (translationMethod > 2 || translationMethod < 0) {
      Logger::getInstance()
          .addWarning("Translation field method not supported.")
          .print();
      return ProcessResult::INVALID_INPUT;
    }

    advectionKernel_.setVelocityField(translationField_);
    advectionKernel_.setIntegrationScheme(
        context.advectionParams.integrationScheme);
    advectionKernel_.setTimeStepRatio(context.advectionParams.timeStepRatio);
    advectionKernel_.setSaveAdvectionVelocities(
        context.advectionParams.velocityOutput);
    advectionKernel_.setDissipationAlpha(
        context.advectionParams.dissipationAlpha);
    advectionKernel_.setIgnoreVoids(context.advectionParams.ignoreVoids);
    advectionKernel_.setCheckDissipation(
        context.advectionParams.checkDissipation);
    // normals vectors are only necessary for analytical velocity fields
    if (translationMethod > 0)
      advectionKernel_.setCalculateNormalVectors(false);

    for (auto &dom : context.domain->getLevelSets()) {
      advectionKernel_.insertNextLevelSet(dom);
    }

    return ProcessResult::SUCCESS;
  }

  void prepareAdvection(const ProcessContext<NumericType, D> &context) {
    // Prepare for advection step
    advectionKernel_.prepareLS();
    context.model->initialize(context.domain, context.processTime);
  }

  ProcessResult performAdvection(ProcessContext<NumericType, D> &context) {
    // Perform the advection step

    if (context.processTime + context.timeStep > context.processDuration) {
      // adjust time step near end
      advectionKernel_.setAdvectionTime(context.processDuration -
                                        context.processTime);
    }

    timer_.start();
    advectionKernel_.apply();
    timer_.finish();

    Logger::getInstance().addTiming("Surface advection", timer_).print();

    if (context.advectionParams.velocityOutput) {
      auto mesh = viennals::Mesh<NumericType>::New();
      viennals::ToMesh<NumericType, D>(context.domain->getLevelSets().back(),
                                       mesh)
          .apply();
      viennals::VTKWriter<NumericType>(
          mesh, "ls_velocities_" + std::to_string(lsVelCounter++) + ".vtp")
          .apply();
    }

    context.timeStep = advectionKernel_.getAdvectedTime();
    context.processTime += context.timeStep;

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

  auto &getTranslationField() const {
    assert(translationField_ != nullptr);
    return translationField_;
  }
};

} // namespace viennaps