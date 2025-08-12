#pragma once

#include "psProcessStrategy.hpp"

namespace viennaps {

template <typename NumericType, int D>
class GeometricProcessStrategy : public ProcessStrategy<NumericType, D> {
public:
  ProcessResult execute(ProcessContext<NumericType, D> &context) override {
    Logger::getInstance().addInfo("Applying geometric model...").print();

    auto geometricModel = context.model->getGeometricModel();
    geometricModel->setDomain(context.domain);
    geometricModel->apply();

    return ProcessResult::SUCCESS;
  }

  std::string getStrategyName() const override {
    return "GeometricProcessStrategy";
  }

  bool canHandle(const ProcessContext<NumericType, D> &context) const override {
    return context.flags.isGeometric;
  }
};

} // namespace viennaps