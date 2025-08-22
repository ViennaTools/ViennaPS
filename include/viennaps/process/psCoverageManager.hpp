#pragma once

#include "psProcessContext.hpp"

namespace viennaps {

template <typename NumericType, int D> class CoverageManager {
public:
  bool initializeCoverages(ProcessContext<NumericType, D> const &context) {
    // Initialize coverage information based on the current context
    auto surfaceModel = context.model->getSurfaceModel();
    assert(surfaceModel != nullptr);
    assert(context.diskMesh != nullptr);

    surfaceModel->initializeCoverages(context.diskMesh->getNodes().size());

    return surfaceModel->getCoverages() != nullptr;
  }

  void convergeCoverages(ProcessContext<NumericType, D> const &context) {
    // Run coverage convergence iterations
  }

  void updateCoverage(ProcessContext<NumericType, D> const &context) {
    // Update coverage information based on the current context
  }

  void outputCoverage() const {
    // Output coverage information
  };
};

} // namespace viennaps