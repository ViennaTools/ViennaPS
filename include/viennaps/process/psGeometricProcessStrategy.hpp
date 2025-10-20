#pragma once

#include "psProcessStrategy.hpp"

namespace viennaps {

template <typename NumericType, int D>
class GeometricProcessStrategy : public ProcessStrategy<NumericType, D> {
public:
  DEFINE_CLASS_NAME(GeometricProcessStrategy)

  ProcessResult execute(ProcessContext<NumericType, D> &context) override {
    Logger::getInstance().addInfo("Applying geometric model...").print();

    if (static_cast<int>(context.domain->getMetaDataLevel()) > 1) {
      context.domain->clearMetaData();
      context.domain->addMetaData(context.model->getProcessMetaData());
    }

    auto geometricModel = context.model->getGeometricModel();
    geometricModel->setDomain(context.domain);
    geometricModel->apply();

    return ProcessResult::SUCCESS;
  }

  bool canHandle(const ProcessContext<NumericType, D> &context) const override {
    return context.flags.isGeometric;
  }
};

} // namespace viennaps