#pragma once

#include <psMaterials.hpp>
#include <psProcessModel.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

namespace IsotropicImplementation {
template <class NumericType, int D>
class IsotropicVelocityField : public psVelocityField<NumericType> {
  const NumericType vel = 1.;
  const std::vector<int> maskMaterials;

public:
  IsotropicVelocityField(const NumericType rate, const std::vector<int> &mask)
      : vel(rate), maskMaterials(mask) {}

  NumericType
  getScalarVelocity(const std::array<NumericType, 3> & /*coordinate*/,
                    int material,
                    const std::array<NumericType, 3> & /* normalVector */,
                    unsigned long /*pointID*/) override {
    if (isMaskMaterial(material)) {
      return 0.;
    } else {
      return vel;
    }
  }

  // the translation field should be disabled when using a surface model
  // which only depends on an analytic velocity field
  int getTranslationFieldOptions() const override { return 0; }

private:
  bool isMaskMaterial(const int material) const {
    for (const auto &mat : maskMaterials) {
      if (material == mat)
        return true;
    }
    return false;
  }
};
} // namespace IsotropicImplementation

/// Isotropic etching with one masking material.
template <typename NumericType, int D>
class psIsotropicProcess : public psProcessModel<NumericType, D> {
public:
  psIsotropicProcess(const NumericType isotropicRate,
                     const psMaterial maskMaterial = psMaterial::None) {
    // default surface model
    auto surfModel = psSmartPointer<psSurfaceModel<NumericType>>::New();

    // velocity field
    std::vector<int> maskMaterialsInt = {static_cast<int>(maskMaterial)};
    auto velField =
        psSmartPointer<IsotropicImplementation::IsotropicVelocityField<
            NumericType, D>>::New(isotropicRate, maskMaterialsInt);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("IsotropicProcess");
  }

  psIsotropicProcess(const NumericType isotropicRate,
                     const std::vector<psMaterial> maskMaterials) {
    // default surface model
    auto surfModel = psSmartPointer<psSurfaceModel<NumericType>>::New();

    // velocity field
    std::vector<int> maskMaterialsInt;
    for (const auto &mat : maskMaterials) {
      maskMaterialsInt.push_back(static_cast<int>(mat));
    }
    auto velField =
        psSmartPointer<IsotropicImplementation::IsotropicVelocityField<
            NumericType, D>>::New(isotropicRate, maskMaterialsInt);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("IsotropicProcess");
  }
};