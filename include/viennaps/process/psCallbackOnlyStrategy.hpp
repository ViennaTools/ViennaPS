#pragma once

#include "psProcessStrategy.hpp"

namespace viennaps {

VIENNAPS_TEMPLATE_ND(NumericType, D)
class CallbackOnlyStrategy : public ProcessStrategy<NumericType, D> {
public:
  DEFINE_CLASS_NAME(CallbackOnlyStrategy)

  ProcessResult execute(ProcessContext<NumericType, D> &context) override {

    auto callback = context.model->getAdvectionCallback();
    callback->setDomain(context.domain);
    callback->applyPreAdvect(0);

    return ProcessResult::SUCCESS;
  }

  bool canHandle(const ProcessContext<NumericType, D> &context) const override {
    return context.processDuration == 0.0 && context.flags.useAdvectionCallback;
  }
};

} // namespace viennaps