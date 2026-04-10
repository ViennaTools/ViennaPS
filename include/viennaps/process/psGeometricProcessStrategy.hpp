#pragma once

#include "psProcessStrategy.hpp"

namespace viennaps {

VIENNAPS_TEMPLATE_ND(NumericType, D)
class GeometricProcessStrategy : public ProcessStrategy<NumericType, D> {
  using GeometricKernel = viennals::GeometricAdvect<NumericType, D>;

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
    if (!mask) {
      // check if mask materials are set, if so create a mask level set from the
      // domain
      auto maskMaterials = geometricModel->getMaskMaterials();
      if (!maskMaterials.empty()) {
        auto domain = context.domain;
        auto materialMap = domain->getMaterialMap();
        if (!materialMap) {
          VIENNACORE_LOG_ERROR("Domain does not have a material map, cannot "
                               "create mask level set "
                               "from mask materials.");
          return ProcessResult::INVALID_INPUT;
        }

        auto maskLevelSet = SmartPointer<viennals::Domain<NumericType, D>>::New(
            domain->getGrid());
        bool foundMaterial = false;

        auto const &levelSets = domain->getLevelSets();
        for (auto &material : maskMaterials) {
          for (int j = 0; j < levelSets.size(); ++j) {
            if (materialMap->getMaterialAtIdx(j) == material) {
              auto lsCopy = SmartPointer<viennals::Domain<NumericType, D>>::New(
                  levelSets[j]);

              // remove all lower level sets that are not mask materials
              for (int k = j - 1; k >= 0; --k) {
                if (MaterialMap::isMaterial(materialMap->getMaterialAtIdx(k),
                                            maskMaterials))
                  continue;

                viennals::BooleanOperation<NumericType, D>(
                    lsCopy, levelSets[k],
                    viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
                    .apply();
              }

              if (foundMaterial) {
                // union with mask level set
                viennals::BooleanOperation<NumericType, D>(
                    maskLevelSet, lsCopy, viennals::BooleanOperationEnum::UNION)
                    .apply();
              } else {
                maskLevelSet = lsCopy;
                foundMaterial = true;
              }
            }
          }
        }

        if (maskLevelSet->getNumberOfPoints() > 0) {
          mask = maskLevelSet;
          if (Logger::hasDebug()) {
            auto dbgMesh = viennals::Mesh<NumericType>::New();
            viennals::ToMesh<NumericType, D>(mask, dbgMesh).apply();
            viennals::VTKWriter<NumericType>(dbgMesh, "geometric_mask_debug")
                .apply();
          }
        } else {
          VIENNACORE_LOG_WARNING(
              "None of the specified mask materials were found in the domain, "
              "cannot create mask level set from mask materials.");
        }
      }
    }

    GeometricKernel(context.domain->getSurface(), dist, mask).apply();

    if (!geometricModel->isDeposition()) {
      // If the top level set was etched, we need to make sure that the
      // other level sets are still consistent by intersecting them with the
      // last level set.
      for (int i = context.domain->getNumberOfLevelSets() - 1; i >= 0; --i) {
        viennals::BooleanOperation<NumericType, D>(
            context.domain->getLevelSets()[i],
            context.domain->getLevelSets().back(),
            viennals::BooleanOperationEnum::INTERSECT)
            .apply();
      }
    }

    return ProcessResult::SUCCESS;
  }

  bool canHandle(const ProcessContext<NumericType, D> &context) const override {
    return context.flags.isGeometric;
  }
};

} // namespace viennaps