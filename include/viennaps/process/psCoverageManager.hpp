#pragma once

#include "psProcessContext.hpp"

namespace viennaps {

template <typename NumericType, int D> class CoverageManager {
public:
  CoverageManager(const ProcessContext<NumericType, D> &context)
      : context_(context) {}

  void updateCoverage() {
    // Update coverage information based on the current context
  }

  void outputCoverage() const {
    // Output coverage information
  }

private:
  ProcessContext<NumericType, D> context_;
};

} // namespace viennaps