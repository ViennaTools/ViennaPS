#pragma once

#include "psProcessStrategy.hpp"

namespace viennaps {

VIENNAPS_TEMPLATE_ND(NumericType, D)
class OxidationStrategy : public ProcessStrategy<NumericType, D> {
public:
  DEFINE_CLASS_NAME(OxidationStrategy)

  ProcessResult execute(ProcessContext<NumericType, D> &context) override {
    context.model->applyModel(context.domain);
    return ProcessResult::SUCCESS;
  }

  bool canHandle(const ProcessContext<NumericType, D> &context) const override {
    return context.model->managesOwnPhysics();
  }
};

} // namespace viennaps
