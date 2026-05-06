#pragma once

#include "psProcessContext.hpp"

namespace viennaps {

VIENNAPS_TEMPLATE_ND(NumericType, D) class CoverageManager {
  std::ofstream covMetricFile_;
  SmartPointer<viennals::PointData<NumericType>> previousCoverages_;

public:
  CoverageManager() = default;
  ~CoverageManager() {
    if (covMetricFile_.is_open())
      covMetricFile_.close();
  }

  bool
  initializeCoverages(ProcessContext<NumericType, D> const &context) const {
    // Initialize coverage information based on the current context
    auto surfaceModel = context.model->getSurfaceModel();
    assert(surfaceModel != nullptr);
    assert(context.diskMesh != nullptr);
    assert(context.diskMesh->getNodes().size() > 0);

    surfaceModel->initializeCoverages(context.diskMesh->getNodes().size());

    return surfaceModel->getCoverages() != nullptr;
  }

  void saveCoverages(ProcessContext<NumericType, D> const &context) {
    previousCoverages_ = SmartPointer<viennals::PointData<NumericType>>::New(
        *context.model->getSurfaceModel()->getCoverages());
  }

  bool
  checkCoveragesConvergence(ProcessContext<NumericType, D> const &context) {

    auto coverages = context.model->getSurfaceModel()->getCoverages();
    assert(previousCoverages_ != nullptr);
    auto deltaMetric =
        calculateCoverageDeltaMetric(coverages, previousCoverages_);
    assert(
        deltaMetric.size() ==
        context.model->getSurfaceModel()->getCoverages()->getScalarDataSize());

    if (Logger::hasInfo()) {
      logMetric(deltaMetric);
      std::stringstream stream;
      stream << std::setprecision(4) << std::fixed;
      stream << "Coverage delta metric: ";
      for (int i = 0; i < coverages->getScalarDataSize(); i++) {
        stream << coverages->getScalarDataLabel(i) << ": " << deltaMetric[i]
               << "\t";
      }
      VIENNACORE_LOG_INFO(stream.str());
    }

    for (auto val : deltaMetric) {
      if (val > context.coverageParams.tolerance)
        return false;
    }
    return true;
  }

private:
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

  void logMetric(const std::vector<NumericType> &metric) {
    if (!Logger::hasDebug())
      return;

    if (!covMetricFile_.is_open()) {
      covMetricFile_.open("coverage_metrics.txt");
    }
    assert(covMetricFile_.is_open());

    for (auto val : metric) {
      covMetricFile_ << val << ";";
    }
    covMetricFile_ << "\n";
  }
};

} // namespace viennaps