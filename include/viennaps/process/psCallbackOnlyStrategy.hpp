#pragma once

#include "psProcessStrategy.hpp"

namespace viennaps {

template <typename NumericType, int D>
class CallbackOnlyStrategy : public ProcessStrategy<NumericType, D> {
public:
  DEFINE_CLASS_NAME(CallbackOnlyStrategy)

  ProcessResult execute(ProcessContext<NumericType, D> &context) override {
    if (context.flags.useAdvectionCallback) {
      auto callback = context.model->getAdvectionCallback();
      callback->setDomain(context.domain);
      callback->applyPreAdvect(0);
    } else {
      Logger::getInstance()
          .addError("No advection callback passed to Process.")
          .print();
    }

    return ProcessResult::SUCCESS;
  }

  bool canHandle(const ProcessContext<NumericType, D> &context) const override {
    return context.processDuration == 0.0 && !context.flags.isGeometric;
  }
};

} // namespace viennaps