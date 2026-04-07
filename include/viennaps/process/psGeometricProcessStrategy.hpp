#pragma once

#include "psProcessStrategy.hpp"

namespace viennaps {

VIENNAPS_TEMPLATE_ND(NumericType, D)
class GeometricProcessStrategy : public ProcessStrategy<NumericType, D> {
public:
  DEFINE_CLASS_NAME(GeometricProcessStrategy)

  ProcessResult execute(ProcessContext<NumericType, D> &context) override {
    VIENNACORE_LOG_INFO("Applying geometric model...");

    if (static_cast<int>(context.domain->getMetaDataLevel()) > 1) {
      context.domain->clearMetaData();
      context.domain->addMetaData(context.model->getProcessMetaData());
    }

    auto geometricModel = context.model->getGeometricModel();
    assert(geometricModel);

    auto dist = geometricModel->getDistribution();
    if (!dist) {
      VIENNACORE_LOG_ERROR(
          "No GeometricAdvectDistribution passed to GeometricModel.");
      return ProcessResult::INVALID_INPUT;
    }

    auto mask = geometricModel->getMask();
    viennals::GeometricAdvect<NumericType, D>(
        context.domain->getLevelSets().back(), dist, mask)
        .apply();

    // Intersect all other level sets with the last one to keep them consistent
    for (int i = context.domain->getNumberOfLevelSets() - 1; i >= 0; --i) {
      viennals::BooleanOperation<NumericType, D>(
          context.domain->getLevelSets()[i],
          context.domain->getLevelSets().back(),
          viennals::BooleanOperationEnum::INTERSECT)
          .apply();
    }

    return ProcessResult::SUCCESS;
  }

  bool canHandle(const ProcessContext<NumericType, D> &context) const override {
    return context.flags.isGeometric;
  }
};

} // namespace viennaps