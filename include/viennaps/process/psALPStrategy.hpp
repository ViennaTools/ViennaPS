#pragma once

#include "psProcessStrategy.hpp"

namespace viennaps {

template <typename NumericType, int D>
class ALPStrategy : public ProcessStrategy<NumericType, D> {
public:
  DEFINE_CLASS_NAME(ALPStrategy)

  ProcessResult execute(ProcessContext<NumericType, D> &context) override {
    return ProcessResult::FAILURE;
  }

  bool canHandle(const ProcessContext<NumericType, D> &context) const override {
    return context.processDuration == 0.0 && !context.flags.isGeometric;
  }
};

} // namespace viennaps