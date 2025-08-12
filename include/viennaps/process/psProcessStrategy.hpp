#pragma once

#include "psProcessContext.hpp"
#include <memory>

namespace viennaps {

enum class ProcessResult {
  SUCCESS,
  INVALID_INPUT,
  EARLY_TERMINATION,
  CONVERGENCE_FAILURE,
  USER_INTERRUPTED
};

template <typename NumericType, int D> class ProcessStrategy {
public:
  virtual ~ProcessStrategy() = default;

  virtual ProcessResult execute(ProcessContext<NumericType, D> &context) = 0;
  virtual std::string getStrategyName() const = 0;
  virtual bool
  canHandle(const ProcessContext<NumericType, D> &context) const = 0;
};

} // namespace viennaps