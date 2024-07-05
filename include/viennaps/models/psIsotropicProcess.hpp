#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"
#include "../psSurfaceModel.hpp"
#include "../psVelocityField.hpp"

#include <vcSmartPointer.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {

template <class NumericType, int D>
class IsotropicVelocityField : public VelocityField<NumericType> {
  const NumericType rate_ = 1.;
  const std::vector<int> maskMaterials_;

public:
  IsotropicVelocityField(NumericType rate, std::vector<int> &&mask)
      : rate_{rate}, maskMaterials_{std::move(mask)} {}

  NumericType getScalarVelocity(const std::array<NumericType, 3> &,
                                int material,
                                const std::array<NumericType, 3> &,
                                unsigned long) override {
    if (isMaskMaterial(material)) {
      return 0.;
    } else {
      return rate_;
    }
  }

  // the translation field should be disabled when using a surface model
  // which only depends on an analytic velocity field
  int getTranslationFieldOptions() const override { return 0; }

private:
  bool isMaskMaterial(const int material) const {
    for (const auto &mat : maskMaterials_) {
      if (material == mat)
        return true;
    }
    return false;
  }
};
} // namespace impl

/// Isotropic etching with one masking material.
template <typename NumericType, int D> class IsotropicProcess {
  SmartPointer<ProcessModel<NumericType, D>> processModel;

public:
  IsotropicProcess(const NumericType isotropicRate,
                   const Material maskMaterial = Material::None) {
    processModel = SmartPointer<ProcessModel<NumericType, D>>::New();

    // default surface model
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    // velocity field
    std::vector<int> maskMaterialsInt = {static_cast<int>(maskMaterial)};
    auto velField =
        SmartPointer<impl::IsotropicVelocityField<NumericType, D>>::New(
            isotropicRate, std::move(maskMaterialsInt));

    processModel->setSurfaceModel(surfModel);
    processModel->setVelocityField(velField);
    processModel->setProcessName("IsotropicProcess");
  }

  IsotropicProcess(const NumericType isotropicRate,
                   const std::vector<Material> maskMaterials) {
    processModel = SmartPointer<ProcessModel<NumericType, D>>::New();

    // default surface model
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    // velocity field
    std::vector<int> maskMaterialsInt;
    for (const auto &mat : maskMaterials) {
      maskMaterialsInt.push_back(static_cast<int>(mat));
    }
    auto velField =
        SmartPointer<impl::IsotropicVelocityField<NumericType, D>>::New(
            isotropicRate, std::move(maskMaterialsInt));

    processModel->setSurfaceModel(surfModel);
    processModel->setVelocityField(velField);
    processModel->setProcessName("IsotropicProcess");
  }

  auto getProcessModel() const { return processModel; }
};

} // namespace viennaps
