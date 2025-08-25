#pragma once

#include "psProcessStrategy.hpp"

namespace viennaps {

template <typename NumericType, int D>
class ALPStrategy : public ProcessStrategy<NumericType, D> {
public:
  DEFINE_CLASS_NAME(ALPStrategy)

  ProcessResult execute(ProcessContext<NumericType, D> &context) override {
    return ProcessResult::NOT_IMPLEMENTED;
  }

  bool canHandle(const ProcessContext<NumericType, D> &context) const override {
    return context.flag.isALP;
  }
};

} // namespace viennaps